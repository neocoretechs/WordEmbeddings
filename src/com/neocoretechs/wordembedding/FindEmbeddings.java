package com.neocoretechs.wordembedding;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;

import com.neocoretechs.relatrix.Result;
import com.neocoretechs.relatrix.Result2;
import com.neocoretechs.rocksack.TransactionId;
import com.neocoretechs.relatrix.client.RelatrixClientTransaction;
import com.neocoretechs.relatrix.client.RelatrixKVClientTransaction;
import com.neocoretechs.relatrix.type.DoubleArray;

/**
 * Operates on the inverted index of Glove50b word embedding vectors stored in Relatrix relationships.<p>
 * This equates to the word mapped to a quantized value of each of the 50 vector elements mapped to the
 * double array of vector values. The search takes the target word, gets the array of embedded values, quantizes them,
 * retrieves each word that is mapped to each quantized value, then does the cosine similarity. This should reduce the
 * search space from over 400k to less than 25k.<p>
 * The purpose is to illustrate Relatix as a vector store that can process embeddings efficiently.<p>
 * Uses cosine similarity.  Euclidean distance or Manhattan distance, may affect the results.
 * @author groff
 *
 */
public class FindEmbeddings {
	//private static RelatrixKVClientTransaction rtc;
	private static RelatrixClientTransaction rtc;
	private static TransactionId xid;
	static long tims = System.currentTimeMillis();
	static int cnt2 = 0;
	
	public FindEmbeddings() {}
	
	@SuppressWarnings("unchecked")
	public static List<String> findClosestEmbeddings(String word, int numResults) throws IOException {
		List<Map.Entry<String, Double>> closestWords = new ArrayList<>();
		//double[] wordVector = (double[]) rtc.get(xid, word);
		//Iterator<?> it = rtc.entrySet(xid,String.class);
		//while(it.hasNext()) {
			//Map.Entry<String,double[]> me = (Map.Entry<String,double[]>) it.next();
			//if(!me.getKey().equals(word)) {
				//double similarity = cosineSimilarity(wordVector, (double[]) me.getValue());
				//closestWords.add(new AbstractMap.SimpleEntry<>((String)me.getKey(), similarity));
			//}
		//}
		//System.out.println("sorting..");
		//closestWords.sort((o1, o) -> o2.getValue().compareTo(o1.getValue()));
		//List<String> result = new A2rrayList<>();
		//for (int i = 0; i < numResults; i++) {
		//	result.add(closestWords.get(i).getKey());
		//}
		//return result;
		// get the vector of the target word. An index to double array vector of embeddings is
		// mapped to each quantized value of every word, so just get the first occurrence
		//Optional<Result> ovec = rtc.findStream(xid, word, '*', '?').findFirst();
		DoubleArray wordVector = null;
		Iterator<?> targit = rtc.findSet(xid, word, '*', '?');
		if(targit.hasNext())
			wordVector = (DoubleArray) ((Result)targit.next()).get();
		//if(ovec.isPresent())
			//wordVector = (DoubleArray) ovec.get().get();
		else {
			System.out.println("Cant locate any occurrance of target word vecors for word:"+word);
			rtc.endTransaction(xid);
			rtc.close();
			System.exit(1);
		}
		// Get relation for each word in inverted index that has a quantized value mapped to an array value of
		// target word. This should ultimately constrain our total result set from 400k words to 25k or less
		for(double dArrayVal: wordVector.get()) {
			int vquant = (int)((dArrayVal + 1 / 2) * 255);
			// get word and double array vector relation that maps to quantized inverted index element
			Iterator<?> it = rtc.findSet(xid, '?', vquant, '?');
			while(it.hasNext()) {
				Result2 res = (Result2) it.next();
				//Jsonb jsonb = JsonbBuilder.create();
				//String result = jsonb.toJson(me);
				//System.out.println(result);
				if(!res.get(0).equals(word)) {
					double similarity = cosineSimilarity(wordVector.get(), ((DoubleArray)res.get(1)).get());
					closestWords.add(new AbstractMap.SimpleEntry<>((String)res.get(0), similarity));
				}
				if((System.currentTimeMillis()-tims) > 5000) {
					System.out.println("Processed "+cnt2+" realtions. Total closest words accumulated="+closestWords.size());
					tims = System.currentTimeMillis();
				}
				++cnt2;
			}
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
		System.out.println("sorting "+closestWords.size());
		closestWords.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
		List<String> result = new ArrayList<>();
		for (int i = 0; i < numResults; i++) {
			result.add(closestWords.get(i).getKey());
		}
		return result;
	}
	
	public static void memoryFind(List<String> words, List<FloatTensor> tensors, String target) {
		int windex = words.indexOf(target);
		if(windex == -1)
			System.exit(1);
		int matches = 0;
		int exceeds = 0;
		double min = 0.0;
		double mincos = 0.0;
		//MinHash th = new MinHash(tensors.get(windex));
		for(int wcnt = 0 ; wcnt < tensors.size(); wcnt++) {
			if(wcnt != windex) {
				//MinHash mh = new MinHash(tensors.get(wcnt));
				//double sim = MinHash.similarity(th.getSignature(), mh.getSignature());
				double sim = -1; // TODO: replace with algo
				double sim2 = FloatTensor.cosineSimilarity(tensors.get(windex),tensors.get(wcnt));
				if(sim > min)
					min = sim;
				if(sim > .8) {
					System.out.println(">.8 match:"+words.get(wcnt)+" and "+words.get(windex)+" index "+wcnt+" ="+sim+" cos:"+sim2);
					++matches;
				} else {
					if(sim > .5)
						System.out.println(">.5 match:"+words.get(wcnt)+" and "+words.get(windex)+" = "+sim+" index "+wcnt+" cos:"+sim2);
				}
				if(sim2 > mincos)
					mincos = sim2;
				if(sim2 > sim) {
					System.out.println("COS exceeded MinHash for:"+words.get(wcnt)+" and "+words.get(windex)+" index "+wcnt+" ="+sim+" cos:"+sim2);
					++exceeds;
				} 
				//System.out.println(Arrays.toString(mh.getSignature()));
			}
		}
		System.out.println(">.8 "+matches+" out of "+tensors.size()+" max best:"+min);
		System.out.println("COS exceeds MinHash "+exceeds+" out of "+tensors.size()+" max best:"+mincos);
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
		rtc = new RelatrixClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		xid = rtc.getTransactionId();
		int numResults = 5;
		List<String> closestWords = findClosestEmbeddings(word, numResults);
		System.out.println("Closest words to '" + word + "':");
		for (String closestWord : closestWords) {
			System.out.println(closestWord);
		}
		rtc.endTransaction(xid);
		rtc.close();
		System.exit(1);
	}
}
