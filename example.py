from tfls import TF_Line_Searcher
import tensorflow as tf
import numpy as np

A = tf.get_variable(name="A", dtype=tf.float64, shape=(64))
loss = tf.reduce_mean(tf.sqrt(tf.abs(A)))
sess = tf.Session()

LS = TF_Line_Searcher(session=sess, loss=loss)

sess.run(tf.global_variables_initializer())
print("Initial loss :%f" % sess.run(loss))

LS.one_step()
print("Loss after one step :%f" % sess.run(loss))
LS.auto()
print("Loss after auto optimiaztion :%f" % sess.run(loss))
del LS

print("\nGradients clipping at Pi/2")
LS = TF_Line_Searcher(session=sess, loss=loss, gradient_alter=lambda x: np.clip(x, -1.57, +1.57))
sess.run(tf.global_variables_initializer())
print("Initial loss :%f" % sess.run(loss))
LS.one_step()
print("Loss after one step :%f" % sess.run(loss))
LS.auto()
print("Loss after auto optimiaztion :%f" % sess.run(loss))
del LS

print("\nApply arctan on Gradients")
LS = TF_Line_Searcher(session=sess, loss=loss, gradient_alter=lambda x: np.arctan(x))
sess.run(tf.global_variables_initializer())
print("Initial loss :%f" % sess.run(loss))
LS.one_step()
print("Loss after one step :%f" % sess.run(loss))
LS.auto()
print("Loss after auto optimiaztion :%f" % sess.run(loss))
del LS

print("\nCheating Gradients")
LS = TF_Line_Searcher(session=sess, loss=loss, gradient_alter=lambda x: 2 / x)
sess.run(tf.global_variables_initializer())
print("Initial loss :%f" % sess.run(loss))
LS.one_step()
print("Loss after one step :%f" % sess.run(loss))
LS.auto()
print("Loss after auto optimiaztion :%f" % sess.run(loss))
del LS

print("\nRun for more steps")
LS = TF_Line_Searcher(session=sess, loss=loss, Xterminate=1e-10, max_step=10000)
sess.run(tf.global_variables_initializer())
print("Initial loss :%f" % sess.run(loss))
LS.one_step()
print("Loss after one step :%f" % sess.run(loss))
LS.auto()
print("Loss after auto optimiaztion :%f" % sess.run(loss))
del LS
