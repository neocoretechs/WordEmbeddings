package com.neocoretechs.wordembedding;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jdk.incubator.vector.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import com.neocoretechs.wordembedding.FloatTensor;
import com.neocoretechs.wordembedding.F32FloatTensor;

import com.neocoretechs.rocksack.TransactionId;

import com.neocoretechs.lsh.Index;
import com.neocoretechs.lsh.RelatrixLSH;

import com.neocoretechs.relatrix.Relatrix;
import com.neocoretechs.relatrix.client.RelatrixClientTransaction;
//import com.neocoretechs.relatrix.client.RelatrixKVClientTransaction;
import com.neocoretechs.relatrix.type.DoubleArray;
import com.neocoretechs.relatrix.type.FloatArray;

/**
 * Load the Glove data file into the K/V store.
 * @author groff
 *
 */
public class LoadWordEmbedding {
	private static final int VECTOR_DIMENSION = 50;
	//GLOVE_FILE = "glove.6B.50d.txt";
	//private static RelatrixKVClientTransaction rtc;
	private static RelatrixClientTransaction rtc;
	private static TransactionId xid;
	private static int COMMITRATE = 1000;
	public static ArrayList<F32FloatTensor> tensors = new ArrayList<F32FloatTensor>();
	public static ArrayList<String> words = new ArrayList<String>();
	public static String embedPath = "D:/etc/Relatrix/db/LSH/Embed";
	
	public LoadWordEmbedding() {}

	private static void loadVectors(String path) throws IOException {
		String line;
		long tims = System.currentTimeMillis();
      	long tim2 = System.currentTimeMillis();
		int cnt2 = 0;
		int cnt = 0;
		/*
		 * MemorySegment and FileChannel, surprisingly slow
	    Arena arena = Arena.ofAuto();
	    try (FileChannel fileChannel = FileChannel.open(FileSystems.getDefault().getPath(path))) {
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size(), arena);
        int index = 0;
        boolean newLine = true;
        StringBuilder sb = new StringBuilder();
		while (index <  fileChannel.size()) {
            char currentChar = (char)tensorData.get(ValueLayout.JAVA_BYTE, index);
            index++;
            if(newLine) {
            	sb = new StringBuilder();
            	newLine = false;
            }
            if (currentChar != '\n') {
            	sb.append(currentChar);
            } else {
            	newLine = true;
    			String[] parts = sb.toString().split(" ");
    			String word = parts[0];
    			//double[] vector = new double[VECTOR_DIMENSION];
    			FloatArray vector = new FloatArray(VECTOR_DIMENSION);
    			ArrayList<Comparable[]> multiStore = new ArrayList<Comparable[]>();
    			for (int i = 0; i < VECTOR_DIMENSION; i++) {
    				//vector[i] = Double.parseDouble(parts[i + 1]);
    				vector.get()[i] = Float.parseFloat(parts[i + 1]);	
    				//Comparable[] c = new Comparable[]{word, vquant, vector};
    				//multiStore.add(c);
    			}
    			F32FloatTensor f32 = new F32FloatTensor(VECTOR_DIMENSION, MemorySegment.ofArray(vector.get()));
    			//System.out.println(f32);
    			++cnt;
    			tensors.add(f32);
    			words.add(word);
            }
            if((System.currentTimeMillis()-tim2) > 5000) {
            	tim2 = System.currentTimeMillis();
            	System.out.println("Loaded "+index+" bytes "+cnt+" vectors in "+(System.currentTimeMillis()-tims)+" ms.");
            }
		}
		}
		*/
		List<String[]> data = FileUtils.readCSVFile(path, " ", -1);
		for(String[] sb: data) {
			String[] parts = sb.toString().split(" ");
			String word = parts[0];
			//double[] vector = new double[VECTOR_DIMENSION];
			FloatArray vector = new FloatArray(VECTOR_DIMENSION);
			ArrayList<Comparable[]> multiStore = new ArrayList<Comparable[]>();
			for (int i = 0; i < VECTOR_DIMENSION; i++) {
				//vector[i] = Double.parseDouble(parts[i + 1]);
				vector.get()[i] = Float.parseFloat(parts[i + 1]);	
				//Comparable[] c = new Comparable[]{word, vquant, vector};
				//multiStore.add(c);
			}
			F32FloatTensor f32 = new F32FloatTensor(VECTOR_DIMENSION, MemorySegment.ofArray(vector.get()));
			//System.out.println(f32);
			++cnt;
			tensors.add(f32);
			words.add(word);
			if((System.currentTimeMillis()-tim2) > 5000) {
				tim2 = System.currentTimeMillis();
				System.out.println("Loaded "+cnt+" vectors in "+(System.currentTimeMillis()-tims)+" ms.");
			}
		}
		/*
			// store inverted index of word, quantized vector element, vector
			rtc.multiStore(xid, multiStore);
			//rtc.store(xid, word, vector);
			if((System.currentTimeMillis()-tims) > 5000) {
				System.out.println("Processed "+cnt2);
				tims = System.currentTimeMillis();
			}
			if(cnt >= COMMITRATE) {
				rtc.commit(xid);
				cnt = 0;
			}
			++cnt2;
			++cnt;
		}
		rtc.commit(xid);
		*/
	}
	
	public static ArrayList<F32FloatTensor> loadTensors(String path) throws IOException {
        loadVectors(path);
		return tensors;
	}

	/**
	 * Command line: Glove data file, local node, remote node, remote port
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		//rtc = new RelatrixKVClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		//rtc = new RelatrixClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		//xid = rtc.getTransactionId();
		Relatrix.setTablespace(embedPath);
		ArrayList<F32FloatTensor> tensors = loadTensors(args[0]);
		RelatrixLSH[] rlsh = new RelatrixLSH[RelatrixLSH.numberOfHashTables];
		for(int i = 0; i < RelatrixLSH.numberOfHashTables; i++)
			rlsh[i] = new RelatrixLSH(i, RelatrixLSH.numberOfHashes, RelatrixLSH.VECTOR_DIMENSION);
		
	}
}
