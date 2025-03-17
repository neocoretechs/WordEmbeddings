package com.neocoretechs.wordembedding;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import com.neocoretechs.relatrix.client.RelatrixClientTransaction;
//import com.neocoretechs.relatrix.client.RelatrixKVClientTransaction;
import com.neocoretechs.relatrix.type.DoubleArray;
import com.neocoretechs.rocksack.TransactionId;
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

	public LoadWordEmbedding() {}

	private static void loadVectors(BufferedReader reader) throws IOException {
		String line;
		long tims = System.currentTimeMillis();
		int cnt2 = 0;
		int cnt = 0;
		while ((line = reader.readLine()) != null) {
			String[] parts = line.split(" ");
			String word = parts[0];
			//double[] vector = new double[VECTOR_DIMENSION];
			DoubleArray vector = new DoubleArray(VECTOR_DIMENSION);
			ArrayList<Comparable[]> multiStore = new ArrayList<Comparable[]>();
			for (int i = 0; i < VECTOR_DIMENSION; i++) {
				//vector[i] = Double.parseDouble(parts[i + 1]);
				vector.get()[i] = Double.parseDouble(parts[i + 1]);
				int vquant = (int)((vector.get()[i] + 1 / 2) * 255);
				Comparable[] c = new Comparable[]{word, vquant, vector};
				multiStore.add(c);
			}
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
	}
	/**
	 * Command line: Glove data file, local node, remote node, remote port
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		//rtc = new RelatrixKVClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		rtc = new RelatrixClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		xid = rtc.getTransactionId();
		try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
			loadVectors(br);
		}

	}
}
