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
top 10 positive theta: ['trumps', 'turnbull', 'trump', 'us', 'says', 'donald', 'korea', 'debate', 'ban', 'north']
top 10 negative theta: ['hillary', 'breaking', 'watch', 'just', 'america', 'victory', 'new', 'are', 'they', 'the']
For fake news:
top 10 positive theta: ['hillary', 'breaking', 'watch', 'just', 'america', 'victory', 'new', 'are', 'they', 'the']
top 10 negative theta: ['trumps', 'turnbull', 'trump', 'us', 'says', 'donald', 'korea', 'debate', 'ban', 'north']
~~~~~~~~~~~~~~~After pruning stopwords~~~~~~~~~~~~~~~
For real news:
top 10 positive theta: ['trumps', 'turnbull', 'trump', 'says', 'donald', 'korea', 'debate', 'ban', 'north', 'comey']
top 10 negative theta: ['hillary', 'breaking', 'watch', 'just', 'america', 'victory', 'new', 'voter', 'star', 'supporter']
For fake news:
top 10 positive theta: ['hillary', 'breaking', 'watch', 'just', 'america', 'victory', 'new', 'star', 'voter', 'supporter']
top 10 negative theta: ['trumps', 'turnbull', 'trump', 'says', 'donald', 'korea', 'debate', 'ban', 'north', 'comey']