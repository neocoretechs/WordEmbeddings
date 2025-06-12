package com.neocoretechs.wordembedding;

import java.util.Comparator;

public class Candidates implements Comparator<Candidates>{
		public String word;
		public double cosDist;
		public FloatTensor tensor;
		@Override
		public String toString() {
			return String.format("Word:%s cos:%.5f" , word, cosDist);
		}
		@Override
		public boolean equals(Object o) {
			return word.equals(((Candidates)o).word);
		}
		@Override
		public int compare(Candidates one, Candidates other) {
			return Double.compare(cosDist,other.cosDist);
		}
}
