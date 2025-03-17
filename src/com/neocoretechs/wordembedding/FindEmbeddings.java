package com.neocoretechs.wordembedding;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.json.bind.Jsonb;
import javax.json.bind.JsonbBuilder;

import com.neocoretechs.relatrix.client.RelatrixKVClientTransaction;
import com.neocoretechs.rocksack.TransactionId;
/**
 * Uses cosine similarity.  Euclidean distance or Manhattan distance, may affect the results.
 * @author groff
 *
 */
public class FindEmbeddings {
	private static RelatrixKVClientTransaction rtc;
	private static TransactionId xid;
	static long tims = System.currentTimeMillis();
	static int cnt2 = 0;
	public FindEmbeddings() {}
	@SuppressWarnings("unchecked")
	public static List<String> findClosestEmbeddings(String word, int numResults) throws IOException {
		List<Map.Entry<String, Double>> closestWords = new ArrayList<>();
		double[] wordVector = (double[]) rtc.get(xid, word);
		
		Iterator<?> it = rtc.entrySet(xid,String.class);
		while(it.hasNext()) {
			Map.Entry<String,double[]> me = (Map.Entry<String,double[]>) it.next();
			Jsonb jsonb = JsonbBuilder.create();
			String result = jsonb.toJson(me);
			System.out.println(result);
			if (!me.getKey().equals(word)) {
				double similarity = cosineSimilarity(wordVector, (double[]) me.getValue());
				closestWords.add(new AbstractMap.SimpleEntry<>((String)me.getKey(), similarity));
			}
			if((System.currentTimeMillis()-tims) > 5000) {
				System.out.println("Processed "+cnt2);
				tims = System.currentTimeMillis();
			}
			++cnt2;
		}
		/*
		rtc.entrySetStream(xid,String.class).forEach(e->{
			Map.Entry<String,double[]> me = (Map.Entry<String,double[]>)e;
			if (!me.getKey().equals(word)) {
				double similarity = cosineSimilarity(wordVector, me.getValue());
				closestWords.add(new AbstractMap.SimpleEntry<>(me.getKey(), similarity));
			}
			if((System.currentTimeMillis()-tims) > 5000) {
				System.out.println("Processed "+cnt2);
				tims = System.currentTimeMillis();
			}
			++cnt2;
		});
		*/
		System.out.println("sorting..");
		closestWords.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
		List<String> result = new ArrayList<>();
		for (int i = 0; i < numResults; i++) {
			result.add(closestWords.get(i).getKey());
		}
		return result;
	}
	
    public static double cosineSimilarity(double[] vector1, double[] vector2) {
        double dotProduct = 0;
        double magnitude1 = 0;
        double magnitude2 = 0;
        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            magnitude1 += Math.pow(vector1[i], 2);
            magnitude2 += Math.pow(vector2[i], 2);
        }
        magnitude1 = Math.sqrt(magnitude1);
        magnitude2 = Math.sqrt(magnitude2);
        return dotProduct / (magnitude1 * magnitude2);
    }
	/**
	 * Command line target word, local node, remote node, remote port
	 * @param args
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception {
		String word = args[0];
		rtc = new RelatrixKVClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		xid = rtc.getTransactionId();
		int numResults = 5;
		List<String> closestWords = findClosestEmbeddings(word, numResults);
		System.out.println("Closest words to '" + word + "':");
		for (String closestWord : closestWords) {
			System.out.println(closestWord);
		}
	}
}
