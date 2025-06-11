package com.neocoretechs.wordembedding;

public class Candidates {
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
}
