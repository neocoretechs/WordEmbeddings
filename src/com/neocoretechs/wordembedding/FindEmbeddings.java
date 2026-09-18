package com.neocoretechs.wordembedding;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import com.neocoretechs.lsh.RelatrixLSH;
import com.neocoretechs.relatrix.AbstractRelation;
import com.neocoretechs.relatrix.Relation;
import com.neocoretechs.relatrix.Relatrix;
import com.neocoretechs.relatrix.RelatrixJson;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.rocksack.TransactionId;
import com.neocoretechs.relatrix.client.json.RelatrixClientJsonTransaction;
import com.neocoretechs.relatrix.key.IndexResolver;
import com.neocoretechs.relatrix.key.NoIndex;
import com.neocoretechs.relatrix.parallel.ExecutionContextHolder;
import com.neocoretechs.relatrix.parallel.ParallelExecutionContext;
import com.neocoretechs.relatrix.type.RelationList;

/**
 * Operates on the inverted index of Glove50b word embedding vectors stored in Relatrix relationships.<p>
 * This equates to the word mapped to a quantized value of each of the 50 vector elements mapped to the
 * double array of vector values. The search takes the target word, gets the array of embedded values, quantizes them,
 * retrieves each word that is mapped to each quantized value, then does the cosine similarity. This should reduce the
 * search space from over 400k to less than 25k.<p>
 * The purpose is to illustrate Relatix as a vector store that can process embeddings efficiently.<p>
 * Uses cosine similarity.  Euclidean distance or Manhattan distance, may affect the results.
 * We store the Float32 tensors using the NoIndex construct in the range value, preventing it from being stored as an instance,
 * as there is no need to query on this value. Consequently no F32FloatTensor table will be constructed.<p>
 * This class retrieves the tensors for a given word index, then performs cosine similarity on the results and writes the collection
 * to a result file. Similarity will be from 1 to -1 with 1 and values near 1 being more similar. A search of the file for cos:0.9 cos:0.8 and cos:0.7 will
 * show the most relevant results. There is no particular order to the results.
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2025
 *
 */
public class FindEmbeddings {
	//private static RelatrixKVClientTransaction rtc;
	private static RelatrixClientJsonTransaction rtc;
	private static TransactionId xid;
	static long tims = System.currentTimeMillis();
	static int cnt2 = 0;
	private static boolean DEBUG = true;
	
	public FindEmbeddings() {}
	
	/**
	 * Command line target word, remote node, remote port<p>
	 * If given just word, 1 param on cmdl, perform embedded db search, this assumes -Dtablespace= is on command line.
	 * @param args word, then optional remove db info
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception {
		if(args.length == 0) {
			System.out.println("Usage:target word [remote node] [remote port]");
			System.exit(1);
		}
		String word = args[0];
		// if we have more than just word
		if(args.length > 1) {
			RelatrixLSH index = null;
			RelationList nearest = null;
			rtc = new RelatrixClientJsonTransaction(args[1],Integer.parseInt(args[2]));
			xid = rtc.getTransactionId();
			Iterator<?> it = rtc.findSet(xid, '*', "has index", '*');
			if(!it.hasNext()) {
				System.out.println("No LSH index...");
				System.exit(1);
			}
			Result res = (Result) it.next();
			index = (RelatrixLSH) res.getRange();
			// now get the tensor with the target word embedding
			it = rtc.findSet(xid, '*', word, '*');
			if(!it.hasNext()) {
				System.out.println("No tensor found for target word "+args[0]);
				rtc.endTransaction(xid);
				System.exit(1);
			}
			res = (Result) it.next();
			int tIndex = (int) Integer.valueOf(res.getDomain().toString());
			if(DEBUG )
				System.out.println("Result range="+res.getRange()+" Class:"+res.getRange().getClass());
			F32FloatTensor tTensor = (F32FloatTensor) res.getRange();
			if(DEBUG )
				System.out.println("Result tensor="+tTensor+" Class:"+tTensor.getClass());
			nearest = index.queryParallel(rtc, xid, tTensor);
			System.out.println("Target word index:"+tIndex+" got nearest:"+nearest.size());
			List<Candidates> candidateList = new ArrayList<Candidates>();
			if(DEBUG )
				System.out.println("Nearest Result range="+((AbstractRelation)nearest.get(0)).getRange()+" Class:"+((AbstractRelation)nearest.get(0)).getRange().getClass());
			for(int i = 0; i  < nearest.size(); i++) {
				Candidates can = new Candidates();
				can.word = (String) ((AbstractRelation)nearest.get(i)).getMap();
				can.tensor = (F32FloatTensor) ((AbstractRelation)nearest.get(i)).getRange();
				can.cosDist = FloatTensor.cosineSimilarity(tTensor, can.tensor);
				int cnt = 0;
				if(!candidateList.contains(can)) {
					candidateList.add(can);
					System.out.print(i+" "+(++cnt)+"\r");
				}
			}
			FileUtils.writeFile(candidateList, word+".txt", false);
			System.out.println("Wrote "+candidateList.size()+" to "+word+".txt");
			rtc.endTransaction(xid);
			//rtc.close();
		} else {
			// embedded
			IndexResolver indexResolver = new IndexResolver();
			ParallelExecutionContext pec = new ParallelExecutionContext(indexResolver, new ConcurrentHashMap<String,Object>());
			ScopedValue.where(ExecutionContextHolder.CONTEXT, pec).run(() -> {
				try {
					Relatrix.getInstance();	
					RelatrixLSH index = null;
					RelationList nearest = null;
					Iterator<?> it = RelatrixJson.findSet('*', "has index", '*');
					if(!it.hasNext()) {
						System.out.println("No LSH index...");
						System.exit(1);
					}
					Result res = (Result) it.next();
					index = (RelatrixLSH) ((Relation)res.get()).getRange();
					// now get the tensor with the target word embedding
					it = Relatrix.findSet('*', word, '*');
					if(!it.hasNext()) {
						System.out.println("No tensor found for target word "+args[0]);
						System.exit(1);
					}
					res = (Result) it.next();
					int tIndex = (int) ((Relation)res.get()).getRange();
					F32FloatTensor tTensor = (F32FloatTensor) ((Relation)res.get()).getDomain();
					nearest = (RelationList) index.queryParallel(tTensor);
					System.out.println("Target word index:"+tIndex);
					List<Candidates> candidateList = new ArrayList<Candidates>();
					for(int i = 0; i  < nearest.size(); i++) {
						Candidates can = new Candidates();
						can.word = (String) ((AbstractRelation)nearest.get(i)).getDomain();
						can.tensor = (F32FloatTensor) ((AbstractRelation)nearest.get(i)).getRange();
						can.cosDist = FloatTensor.cosineSimilarity(tTensor, can.tensor);
						int cnt = 0;
						if(!candidateList.contains(can)) {
							candidateList.add(can);
							System.out.print(i+" "+(++cnt)+"\r");
						}
					}
					FileUtils.writeFile(candidateList, word+".txt", false);
					System.out.println("Wrote "+candidateList.size()+" to "+word+".txt");
				} catch (IllegalAccessException | ClassNotFoundException | IOException e) {
					e.printStackTrace();
					System.exit(1);
				}
			});
		}
		System.exit(1);
	}

}
