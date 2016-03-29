
sgd = SGD(lr=1, decay=1e-1, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

for i in xrange(10):
    model.train_on_batch(x[i*16:(i+1)*16], y[(i)*16:(i+1)*16], accuracy=True)
    print model.evaluate(x_test, y_test, batch_size=4, show_accuracy=True)
    print model.predict(x_test, batch_size=4)

