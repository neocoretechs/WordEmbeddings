
<h2> Word Embeddings Relatrix Storage</h2>
An Index contains one or more locality sensitive hash tables. These hash tables contain the mapping between a combination of a number of hashes
(encoded using an integer) and a list of possible nearest neighbors.<p>

 A hash function can hash a vector of arbitrary dimensions to an integer
 representation. The hash function needs to be locality sensitive to work in
 the locality sensitive hash scheme. Meaning that vectors that are 'close'
 according to some metric have a high probability to end up with the same
 hash.<p>
 In the context of Locality-Sensitive Hashing (LSH), w represents the bucket width or window size.<p>
 When we compute the hash value for a vector using a random projection. Here's what each component does:<p>
 vector.dot(randomProjection): Computes the dot product of the input vector and a random projection vector. <br>
 This projects the input vector onto a random direction.<br>
 offset: Adds a random offset to the projected value. <br>
 This helps to shift the projected values and create a more uniform distribution.<p>
 w: The bucket width or window size. This value determines the granularity of the hash function.<br>
 By dividing the projected value (plus offset) by w, you're essentially:<br>
 Quantizing the projected values into discrete buckets.<br>
 Assigning each bucket a unique hash value. <br>
 The choice of w affects the trade-off between:<br>
 Precision: Smaller w values result in more precise hashing, but may lead to more collisions.
 Larger w values result in fewer collisions, but may reduce precision.<p>
 In general, w is a hyperparameter that needs to be tuned for specific applications and datasets. 
 A good choice of w can significantly impact the performance of the LSH algorithm.<p>
 This codebase stores embeddings in the Relatrix database to serve as a template for encoding and retrieving
 a given set of floating point tensors. The dataset in question is the Glove6B dataset which contains 400000
 entries of words followed by the embedding tensor with 50 to 100 elements depending on the file chosen. We default to the
 50D dataset, although the VECTOR_DIMENSION field can be changed for the other datasets. 400k entries with dimension 50
 storing an LSH index of 16 hashes results in quite a large dataset, on the order of 4 to 5 gigabytes with many millions of relationships.
 Two possible modes are supported:<p>
 A transactional database server with a timed commit of 10 seconds is used, and the system uses a client to that remote server,
 which must be started before the example can begin. The com.neocoretechs.relatrix.server.json.RelatrixTransactionServerJson process must be started
 on a given remote node and port, with the -Dtablespace= property set for the location of the resulting databases.<p>
 if the remote node and port arguments are left out on the command line, an embedded database will be used, again controlled by the -Dtablespace environment,
 and in this case no remote server process need be started, but care must be taken to allow sufficient heap size etc.<p>
 Once LoadWordEmbedding has completed, FindEmbeddings can be used to retrieve similarities to a given word from the dataset.
