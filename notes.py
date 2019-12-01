# Turn down for faster convergence
t0 = time.time()
train_size = 50000
test_size = 10000

### load MNIST data from https://www.openml.org/d/554
X_org, y_org = fetch_openml('mnist_784', version=1, return_X_y=True)

X_conv = np.zeros((70000, 196*2))
for i, x in enumerate(X_org):
    X_h = np.real(signal.convolve2d(x.reshape(28,28), horizn, boundary='symm', mode='same'))
    X_v = np.real(signal.convolve2d(x.reshape(28,28), vertic, boundary='symm', mode='same'))
    X_conv[i] = np.r_[MAX_pool(X_h).flatten(), MAX_pool(X_v).flatten()]

plt.imshow(X_conv[2].reshape(28,14))
plt.show()

y_org[2]

# shuffle data
X = X_conv
y = y_org
X = X_conv.reshape((X_conv.shape[0], -1))

# pick training and test data sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,test_size=test_size)

# scale data to have zero mean and unit variance [required by regressor]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# apply logistic regressor with 'sag' solver, C is the inverse regularization strength
clf = LogisticRegression(C=1e5,
                         multi_class='multinomial',
                         penalty='l2', solver='sag', tol=0.1)
# fit data
clf.fit(X_train, y_train)
# percentage of nonzero weights
sparsity = np.mean(clf.coef_ == 0) * 100
# compute accuracy
score = clf.score(X_test, y_test)

#display run time
run_time = time.time() - t0
print('Example run in %.3f s' % run_time)

print("Sparsity with L2 penalty: %.2f%%" % sparsity)
print("Test score with L2 penalty: %.4f" % score)
