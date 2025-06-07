package com.neocoretechs.wordembedding;

import java.io.Externalizable;
import java.util.Arrays;
import java.util.concurrent.atomic.DoubleAdder;

import jdk.incubator.vector.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
* Over-simplified, shapeless, float tensor.
* <p>
* Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
* e.g. can represent a sequence of quantized floats.
*/
public abstract class FloatTensor implements Externalizable, Comparable {
	    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
	    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

	    static short readShort(MemorySegment memorySegment, long offset) {
	        return memorySegment.get(ValueLayout.JAVA_SHORT, offset);
	        //return UNSAFE.getShort(memorySegment.address() + offset);
	    }
	    
	    static int readInt(MemorySegment memorySegment, long offset) {
	        return memorySegment.get(ValueLayout.JAVA_INT, offset);
	        //return UNSAFE.getShort(memorySegment.address() + offset);
	    }
	    
	    static float readFloat(MemorySegment memorySegment, long offset) {
	        return memorySegment.get(ValueLayout.JAVA_FLOAT, offset);
	        //return UNSAFE.getShort(memorySegment.address() + offset);
	    }
	    
	    static byte readByte(MemorySegment memorySegment, long offset) {
	        return memorySegment.get(ValueLayout.JAVA_BYTE, offset);
	        //return UNSAFE.getByte(memorySegment.address() + offset);
	    }

	    // Preferred vector size for the fast multiplication routines.
	    // (Apple Silicon) NEON only supports up-to 128bit vectors.
	    static final VectorSpecies<Float> F_SPECIES;
	    static final VectorSpecies<Integer> I_SPECIES;
	    static final VectorSpecies<Short> S_SPECIES_HALF;

	    static {
	        if (USE_VECTOR_API) {
	            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
	            I_SPECIES = F_SPECIES.withLanes(int.class);
	            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
	            assert F_SPECIES.length() == S_SPECIES_HALF.length();
	        } else {
	            F_SPECIES = null;
	            I_SPECIES = null;
	            S_SPECIES_HALF = null;
	        }
	    }

	    public abstract int size();

	    public abstract float getFloat(int index);

	    public abstract void setFloat(int index, float value);

	    public abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

	    public static int numberOfElements(int... dimensions) {
	        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
	        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
	    }

	    public static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
	        float result = 0f;
	        for (int j = 0; j < size; j++) {
	            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
	        }
	        return result;
	    }

	    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
	        return scalarDot(this, thisOffset, that, thatOffset, size);
	    }

	    public void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
	        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
	    }

	    public void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
	        if (that.length != out.length) {
	            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
	        }
	        Parallel.parallelForLong(0, dim0 * context, ti -> {
	            int idxArr = (int) (ti / dim0);
	            int i = (int) (ti % dim0);
	            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1)); 
	        });
	    }

	    @FunctionalInterface
	    interface AggregateFunction {
	        float apply(float acc, float value);
	    }

	    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
	        float result = seed;
	        for (int i = 0; i < size; ++i) {
	            result = reduce.apply(result, getFloat(thisOffset + i));
	        }
	        return result;
	    }

	    float sum(int thisOffset, int size) {
	        return reduce(thisOffset, size, 0f, Float::sum);
	    }

	    float max(int thisOffset, int size) {
	        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
	    }

	    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
	        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
	    }

	    int argmax(int thisOffset, int size) {
	        assert size > 0;
	        int maxIndex = thisOffset;
	        float maxValue = this.getFloat(maxIndex);
	        int endIndex = thisOffset + size;
	        for (int i = thisOffset; i < endIndex; ++i) {
	            float f = this.getFloat(i);
	            if (f > maxValue) {
	                maxValue = f;
	                maxIndex = i;
	            }
	        }
	        return maxIndex;
	    }

	    int argmax() {
	        return argmax(0, size());
	    }

	    @FunctionalInterface
	    interface MapFunction {
	        float apply(float value);
	    }

	    @FunctionalInterface
	    interface MapWithIndexFunction {
	        float apply(float value, int index);
	    }

	    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
	        int endIndex = thisOffset + size;
	        for (int i = thisOffset; i < endIndex; ++i) {
	            setFloat(i, mapFunction.apply(getFloat(i)));
	        }
	        return this;
	    }

	    FloatTensor mapInPlace(MapFunction mapFunction) {
	        return mapInPlace(0, size(), mapFunction);
	    }

	    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
	        int endOffset = thisOffset + size;
	        for (int i = thisOffset; i < endOffset; ++i) {
	        	//System.out.println("setFloat:"+i+" of size:"+size);
	            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
	        }
	        return this;
	    }

	    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
	        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
	    }

	    FloatTensor addInPlace(FloatTensor that) {
	        return addInPlace(0, that, 0, size());
	    }

	    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
	        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
	    }

	    FloatTensor multiplyInPlace(FloatTensor that) {
	        return multiplyInPlace(0, that, 0, size());
	    }

	    FloatTensor divideInPlace(int thisOffset, int size, float value) {
	        return mapInPlace(thisOffset, size, f -> f / value);
	    }

	    FloatTensor fillInPlace(int thisOffset, int size, float value) {
	        return mapInPlace(thisOffset, size, unused -> value);
	    }

	    FloatTensor softmaxInPlace(int thisOffset, int size) {
	        // find max value (for numerical stability)
	        float maxVal = max(thisOffset, size);
	        // exp and sum
	        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
	        float sum = sum(thisOffset, size);
	        // normalize
	        return divideInPlace(thisOffset, size, sum);
	    }

	    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
	        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
	        for (int i = 0; i < size; ++i) {
	            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
	        }
	        return this;
	    }
	    
	    public static float cosineSimilarity(FloatTensor a, FloatTensor b) {
	    	float dotProduct = a.dot(0, b, 0, a.size());
	    	DoubleAdder aNormAdder = new DoubleAdder();
	    	DoubleAdder bNormAdder = new DoubleAdder();
	    	Parallel.parallelFor(0, a.size(), t -> {
	    	    aNormAdder.add(a.getFloat(t) * a.getFloat(t));
	    	    bNormAdder.add(b.getFloat(t) * b.getFloat(t));
	    	});
	    	float aNorm = (float) Math.sqrt(aNormAdder.sum());
	    	float bNorm = (float) Math.sqrt(bNormAdder.sum());
	    	return (dotProduct / (aNorm * bNorm));
	    }
	    
	    public void verify() {
	    	System.out.println("size:"+size());
	      	System.out.println("Verified via String of length:"+toString().length());
	    }
	    
	    public String toString() {
	    	StringBuilder sb = new StringBuilder("[");
	    	for(int i = 0; i < size(); i++) {
	    		sb.append(getFloat(i));
	    		if(i == (size()-1)) 
	    			sb.append("]");
	    		else
	    			sb.append(",");
	    	}
	    	return sb.toString();
	    }
}
