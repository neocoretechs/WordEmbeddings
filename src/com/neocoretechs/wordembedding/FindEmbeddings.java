package com.neocoretechs.wordembedding;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import com.neocoretechs.lsh.RelatrixLSH;

import com.neocoretechs.relatrix.Relatrix;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.rocksack.TransactionId;
import com.neocoretechs.relatrix.client.RelatrixClientTransaction;

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
	
	/**
	 * Command line target word, local node, remote node, remote port
	 * @param args
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception {
		//String word = args[0];
		//rtc = new RelatrixClientTransaction(args[1],args[2],Integer.parseInt(args[3]));
		//xid = rtc.getTransactionId();
		//int numResults = 5;
		//ArrayList<F32FloatTensor> tensors = loadTensors(args[0]);
		//List<String> closestWords = findClosestEmbeddings(word, numResults);
		//System.out.println("Closest words to '" + word + "':");
		//for (String closestWord : closestWords) {
		//	System.out.println(closestWord);
		//}
		//rtc.endTransaction(xid);
		//rtc.close();
		Relatrix.setTablespace(LoadWordEmbedding.embedPath);
		RelatrixLSH index = null;
		List<Result> nearest = null;
		try {
			Iterator<?> it = Relatrix.findSet('*', "has index", '?');
			if(!it.hasNext()) {
				System.out.println("No LSH index...");
				System.exit(1);
			}
			Result res = (Result) it.next();
			index = (RelatrixLSH) res.get();
			// now get the tensor with the target word embedding
			it = Relatrix.findSet('?',  args[0], '?');
			if(!it.hasNext()) {
				System.out.println("No tensor found for target word "+args[0]);
				System.exit(1);
			}
			res = (Result) it.next();
			int tIndex = (int) res.get(0);
			F32FloatTensor tTensor = (F32FloatTensor) res.get(1);
			nearest = index.queryParallel(tTensor);
			System.out.println("Target word index:"+tIndex);
			List<Candidates> candidateList = new ArrayList<Candidates>();
			for(int i = 0; i  < nearest.size(); i++) {
				Candidates can = new Candidates();
				can.word = (String) nearest.get(i).get(0);
				can.tensor = (FloatTensor) nearest.get(i).get(1);
				can.cosDist = FloatTensor.cosineSimilarity(tTensor, can.tensor);
				int cnt = 0;
				if(!candidateList.contains(can)) {
					candidateList.add(can);
					System.out.print(i+" "+(++cnt)+"\r");
				}
			}
			FileUtils.writeFile(candidateList, args[0]+".txt", false);
		} catch (IllegalAccessException | ClassNotFoundException | IOException e) {
				e.printStackTrace();
				System.exit(1);
		}
		System.exit(1);
	}

}
