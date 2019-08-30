import time
import numpy as np
import scipy.stats as st

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def train_model(X_train, y_train, X_val, y_val, train_fn, val_fn, num_epochs,
                batch_size=128, comp_acc=False, smart_stop=False):
    print("Starting training...")
    quit = False
    if smart_stop:
        smart_buff_size = 10
        smart_tol = 1e-3
        smart_buffer = np.array([])
    if comp_acc:
        best_val = float('inf')
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,
                                         batch_size, shuffle=True):
            inputs, targets = batch
            try:
                train_err += train_fn(inputs, targets)
            except KeyboardInterrupt:
                quit = True
                break
            train_batches += 1
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val,
                                         batch_size, shuffle=False):
            inputs, targets = batch
            if comp_acc:
                err, acc = val_fn(inputs, targets)
                val_acc += acc
            else:
                err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        if comp_acc:
            print("  validation CE:\t\t{:.2f}".format(
                  val_acc / val_batches))
            if val_acc / val_batches < best_val:
                best_val = val_acc / val_batches

        if smart_stop:
            smart_buffer = np.append(smart_buffer, val_err)
            if smart_buffer.shape[0] > smart_buff_size:
                smart_buffer = smart_buffer[-smart_buff_size:]
                out = st.linregress(np.arange(smart_buff_size), smart_buffer)
                slope = out[0]
                print("Condition", slope, -smart_tol * val_err)
                if slope > -smart_tol * val_err:
                    print("Smart stop.")
                    quit = True
                    break

        if np.isnan(train_err) or quit:
            break
    if smart_stop:
        return smart_buffer.mean()
    if val_acc:
        return best_val
