package com.neocoretechs.wordembedding;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.lang.foreign.MemorySegment;

import com.neocoretechs.rocksack.TransactionId;

import com.neocoretechs.lsh.RelatrixLSH;
import com.neocoretechs.relatrix.DuplicateKeyException;
import com.neocoretechs.relatrix.Relatrix;
import com.neocoretechs.relatrix.client.json.RelatrixClientJsonTransaction;
//import com.neocoretechs.relatrix.client.RelatrixKVClientTransaction;
import com.neocoretechs.relatrix.key.IndexResolver;
import com.neocoretechs.relatrix.parallel.ExecutionContextHolder;
import com.neocoretechs.relatrix.parallel.ParallelExecutionContext;
import com.neocoretechs.relatrix.type.FloatArray;

/**
 * Load the Glove data file into the Relatrix store.<p>
 * Operates on the inverted index of Glove50b word embedding vectors stored in Relatrix relationships.<p>
 * This equates to the word mapped to a quantized value of each of the 50 vector elements mapped to the
 * double array of vector values. The search takes the target word, gets the array of embedded values, quantizes them,
 * retrieves each word that is mapped to each quantized value, then does the cosine similarity. This should reduce the
 * search space from over 400k to less than 25k.<p>
 * The purpose is to illustrate Relatix as a vector store that can process embeddings efficiently.<p>
 * Uses cosine similarity.  Euclidean distance or Manhattan distance, may affect the results.
 * We store the Float32 tensors using the NoIndex construct in the range value, preventing it from being stored as an instance,
 * as there is no need to query on this value. Consequently no F32FloatTensor table will be constructed.<p>
 * Designed to read the Glove6B dataset which includes vectors of 50 to 100 elements. The value of the field
 * VECTOR_DIMENSION can be changed for the different datasets. Default is 50. A timed commit of 10 seconds is
 * performed on the remote Relatrix JSON transaction database server. com.neocoretechs.relatrix.server.json.RelatrixTransactionServerJson
 * process must be started on a remote node and port with the -Dtablespace= environment variable set on the command line, or
 * if the remote node and port arguments are left out, an embedded database will be used, again controlled by the -Dtablespace environment
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2025
 *
 */
public class LoadWordEmbedding {
	private static final int VECTOR_DIMENSION = 50;
	//GLOVE_FILE = "glove.6B.50d.txt";
	//private static RelatrixKVClientTransaction rtc;
	private static RelatrixClientJsonTransaction rtc;
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
			//String[] parts = sb.toString().split(" ");
			String word = sb[0];
			//double[] vector = new double[VECTOR_DIMENSION];
			FloatArray vector = new FloatArray(VECTOR_DIMENSION);
			ArrayList<Comparable[]> multiStore = new ArrayList<Comparable[]>();
			for (int i = 0; i < VECTOR_DIMENSION; i++) {
				//vector[i] = Double.parseDouble(parts[i + 1]);
				vector.get()[i] = Float.parseFloat(sb[i + 1]);	
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
	 * @throws DuplicateKeyException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 */
	public static void main(String[] args) throws IOException, IllegalAccessException, ClassNotFoundException, DuplicateKeyException {
		if(args.length == 0) {
			System.out.println("Usage:target path [remote node] [remote port]");
			System.exit(1);
		}
		String word = args[0];
		ArrayList<F32FloatTensor> tensors = loadTensors(word);
		RelatrixLSH rlsh = new RelatrixLSH(RelatrixLSH.numberOfHashes, RelatrixLSH.numberOfHashTables, RelatrixLSH.VECTOR_DIMENSION);
		// if we have more than just word
		if(args.length > 1) {
			rtc = new RelatrixClientJsonTransaction(args[1],Integer.parseInt(args[2]));
			xid = rtc.getTransactionId();
			try {
				rtc.store(xid, rlsh.getKey(), "has index", rlsh);
			} catch (IOException e) {
				e.printStackTrace();
				rtc.endTransaction(xid);
				System.exit(1);
			}
			long tims = System.currentTimeMillis();
			long tim2 = System.currentTimeMillis();
			for(int i = 0; i < tensors.size(); i++) {
				rlsh.add(rtc, xid, words.get(i), tensors.get(i));
				if((System.currentTimeMillis()-tim2) > 10000) {
					tim2 = System.currentTimeMillis();
					System.out.println("Loaded "+i+" vectors in "+(System.currentTimeMillis()-tims)+" ms.");
					rtc.commit(xid);
				}
			}
			rtc.commit(xid);
			rtc.endTransaction(xid);
		} else {
			IndexResolver indexResolver = new IndexResolver();
			ParallelExecutionContext pec = new ParallelExecutionContext(indexResolver, new ConcurrentHashMap<String,Object>());
			ScopedValue.where(ExecutionContextHolder.CONTEXT, pec).run(() -> {
				try {
					Relatrix.store(rlsh.getKey(), "has index", rlsh);
				} catch (IllegalAccessException | ClassNotFoundException | IOException | DuplicateKeyException e) {
					e.printStackTrace();
				}
				long tims = System.currentTimeMillis();
				long tim2 = System.currentTimeMillis();
				for(int i = 0; i < tensors.size(); i++) {
					rlsh.add(words.get(i), tensors.get(i));
					if((System.currentTimeMillis()-tim2) > 10000) {
						tim2 = System.currentTimeMillis();
						System.out.println("Loaded "+i+" vectors in "+(System.currentTimeMillis()-tims)+" ms.");
					}
				}
			});
		}
	}
}
