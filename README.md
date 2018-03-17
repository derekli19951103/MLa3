# MLa3
part7a splitter and maxdepth:
best maxdepth for best splitter: 200
its validating performance: 0.787755102041
best maxdepth for random splitter: 400
its validating performance: 0.779591836735


part 4 regularization:
============Trained using training set, test using testing set============
L2 Regularization accuracy: 0.828220858896
L1 Regularization accuracy: 0.820040899796
size/parameter inverse relation

part 6ab:
For real news:
top 10 positive theta: ['trumps', 'turnbull', 'us', 'trump', 'says', 'donald', 'korea', 'debate', 'north', 'ban']
top 10 negative theta: ['hillary', 'watch', 'breaking', 'just', 'america', 'victory', 'new', 'are', 'the', 'they']
For fake news:
top 10 positive theta: ['trumps', 'turnbull', 'us', 'trump', 'says', 'donald', 'korea', 'debate', 'north', 'ban']
top 10 negative theta: ['hillary', 'watch', 'breaking', 'just', 'america', 'victory', 'new', 'are', 'the', 'they']
~~~~~~~~~~~~~~~After pruning stopwords~~~~~~~~~~~~~~~
For real news:
top 10 positive theta: ['trumps', 'turnbull', 'trump', 'says', 'donald', 'korea', 'debate', 'north', 'ban', 'comey']
top 10 negative theta: ['hillary', 'watch', 'breaking', 'just', 'america', 'victory', 'new', 'voter', 'star', 'supporter']
For fake news:
top 10 positive theta: ['trumps', 'turnbull', 'trump', 'says', 'donald', 'korea', 'debate', 'north', 'ban', 'comey']
top 10 negative theta: ['hillary', 'watch', 'breaking', 'just', 'america', 'victory', 'new', 'voter', 'star', 'supporter']