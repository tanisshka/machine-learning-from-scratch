import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#Initialization
x=tf.constant(4)
print(x)

y=tf.constant(5,shape=(1,1),dtype=float)
print(y)

z=tf.constant([[1,2,3],[4,5,6]])
print(z)

w=tf.ones(shape=(3,3))
print(w)

u=tf.eye(3) #Creates an idetity matrix of 3 x  3
print(u)

a=tf.random.normal((3,3),mean=0,stddev=1)
print(a)

b=tf.random.uniform((1,3),minval=0,maxval=1)
print(b)

c=tf.range(start=1, limit=10, delta=2)
print(c)

#Mathematical operations
e=tf.constant([1,2,3])
f=tf.constant([4,5,6])

print(tf.add(e,f))
print(tf.subtract(e,f))
print(tf.multiply(e,f))
print(tf.divide(e,f))


#Indexing
h=tf.constant([0,1,2,3,4,5,6,7,8,9])
print(h[:])
print(h[::2])
print(h[::-1])