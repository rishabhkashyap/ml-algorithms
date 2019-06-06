import  numpy as np
import matplotlib.pyplot as plt

n_class=500

X1=np.random.randn(n_class,2)+np.array([0,-2])
X2=np.random.randn(n_class,2)+np.array([2,2])
X3=np.random.randn(n_class,2)+np.array([-2,2])


#Concatinate X1,X2 and X3 on rows,resulting matrix will be of size 1500X2
X=np.vstack([X1,X2,X3])
y=np.array([0]*n_class+[1]*n_class+[2]*n_class)
D=2
M=3
K=3

W1=np.random.randn(D,M)
b1=np.random.randn(M)
W2=np.random.randn(M,K)
b2=np.random.randn(K)

def feed_forward(X,W1,b1,W2,b2):
    Z=1/1+np.exp(-X.dot(W1)-b1)
    A=Z.dot(W2)+b2
    expA=np.exp(A)
    Y=expA/np.sum(expA,axis=1,keepdims=True)
    return Y


def classification_rate(y,P):

    n_class=0
    n_total=0
    for i in range(len(y)):
        n_total+=1
        if(y[i]==1):
            n_class+=1


    return n_class/n_total


if __name__=='__main__':
    p_y_given_x=feed_forward(X,W1,b1,W2,b2)
    p=np.argmax(p_y_given_x,axis=1)
    plt.scatter(X[:,0],X[:,1],c=y,alpha=0.5)
    plt.show()
    print('classification rate of randomly choosen weight = {}'.format(classification_rate(y,p)))


