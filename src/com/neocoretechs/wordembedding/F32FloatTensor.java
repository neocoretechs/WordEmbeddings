package com.neocoretechs.wordembedding;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

import jdk.incubator.vector.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class F32FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

	int size;
	transient MemorySegment memorySegment;
	
	public F32FloatTensor() {}
	
	public F32FloatTensor(int size, MemorySegment memorySegment) {
		this.size = size;
		this.memorySegment = memorySegment;
	}

	@Override
	public int size() {
		return size;
	}

	@Override
	public float getFloat(int index) {
		assert 0 <= index && index < size;
		return readFloat(memorySegment, index * 4);
	}

	@Override
	public void setFloat(int index, float value) {
		throw new UnsupportedOperationException("setFloat");	
	}

	@Override
	public FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
		 throw new UnsupportedOperationException("getFloatVector");
	}


	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		size = in.readInt();
		long bs = in.readLong();
		memorySegment = Arena.ofAuto().allocate(bs, 1);
		for(int i = 0; i < bs; i++)
			memorySegment.set(ValueLayout.JAVA_BYTE, i, (byte)(in.read() & 0xFF));
	}

	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((F32FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((F32FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

