# Day_01_07_basic_tf.py
import tensorflow as tf

def not_used():
    a = tf.constant(3)
    b = tf.Variable(5)

    print(a)
    print(b)

    sess = tf.InteractiveSession()
    b.initializer.run()
    print(a.eval())
    print(b.eval())
    sess.close()


def not_used_2():
    a = tf.constant(3)
    b = tf.Variable(5)
    c = a + b
    d = tf.add(a, b)

    sess = tf.Session()
    b.initializer.run(session=sess)

    # print(c.eval())
    print(sess.run(c))
    print(sess.run(d))

    sess.close()


one = tf.constant(1)
value = tf.Variable(0)
update = tf.add(value, one)         # value + one
assign = tf.assign(value, update)   # value = update

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(3):
    # print(sess.run(value))
    # print(sess.run(assign),
    #       sess.run(value),
    #       sess.run(update))
    print(sess.run(update),
          sess.run(assign),
          sess.run(value))

sess.close()
