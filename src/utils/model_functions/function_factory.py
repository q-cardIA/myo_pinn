# import necessary modules
import numpy as np
import tensorflow as tf


def function_factory(model, loss, train_x, train_y, pars):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # stitch and partition indexes
    count = 0
    stitch = []
    part = []

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        stitch.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # will be returned by this factory
    @tf.function
    def f(params_1d):
        with tf.GradientTape() as g:
            assign_new_model_parameters(params_1d)
            # loss_value = loss(train_y, model(train_x[0], train_x[1]))
            loss_value = loss(train_x[0], train_x[1], train_y)

        # calculate gradients and convert to 1D tf.Tensor
        grads = g.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(stitch, grads)
        error = model.get_error(pars)

        # track
        f.iter.assign_add(1)
        if f.iter % 100 == 0 or f.iter == 1:
            tf.print("Epoch:", f.iter, "loss:", loss_value, "error:", error)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = stitch
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    return f
