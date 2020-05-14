import numpy as np

"""
Predicts the target values for data in the file at 'test_X_file_path'
Writes the predicted values to the file named "predicted_test_Y.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""

def preprocessing(s):
    """
    1. Remove the characters which are not alphabets
    2. Replace multiple spaces with single space
    3. Convert the string to lowercase
    """
    print("preprocessing")
    flag=0
    s1=""
    for i in range(0,len(s),1):
        if(ord(s[i])>=65 and ord(s[i])<=90):
            if(flag==1 and len(s1)>0):
                s1=s1+" "
            s1=s1+chr(ord(s[i])+32)
            flag=0
        elif(ord(s[i])>=97 and ord(s[i])<=122):
            if(flag==1 and len(s1)>0):
                s1=s1+" "
            s1=s1+s[i]
            flag=0
        elif(s[i]==' '):
            flag=1
        else:
            flag=1
    return s1
        

def class_wise_words_frequency_dict(X, Y):
    """
    Compute classwise words frequence dict
    Return a dict with key as class and value as a dict. 
    The value dict contains words as key and its frequency in documents of class as value
    """
    print("words")
    d={}
    for i in range(0,X.shape[0],1):
        if(Y[i] in d.keys()):
            for j in X[i].split(" "):
                if(j in d[Y[i]].keys()):
                    d[Y[i]][j]+=1
                else:
                    d[Y[i]][j]=1
        else:
            d[Y[i]]={}
            for j in X[i].split(" "):
                if(j in d[Y[i]].keys()):
                    d[Y[i]][j]+=1
                else:
                    d[Y[i]][j]=1
    return d

def compute_prior_probabilities(Y):
    """
    Compute the prior probabilites of each class
    Return a dict of classes and the corresponding prior probabilities
    """
    print("prior")
    d={}
    u=np.unique(Y,return_counts=True)
    s=np.sum(u[1])
    for i in range(0,len(u[0]),1):
        d[u[0][i]]=u[1][i]/s
    return d

def get_class_wise_denominators_likelihood(classwise_words_dict,alpha):
    """
    Given list of strings and the corresponding classes,
    Compute the denominator of likelihood (add-one laplace smoothing)
    Return a dict of classes and the corresponding likelihood denominator values
    """
    print("denominators")
    denominator_d={}
    t_s=0
    for i in classwise_words_dict:
        s=0
        for j in classwise_words_dict[i]:
            s=s+classwise_words_dict[i][j]
            t_s=t_s+1
        denominator_d[i]=s
    for i in denominator_d:
        denominator_d[i]=denominator_d[i]+alpha*t_s
    return denominator_d

def compute_likelihood(test_X, c,classwise_words_dict,classwise_denominators_dict,alpha):
    """
    Complete the function using the return values of train function class_wise_frequency_dict, class_wise_denominators, prior_probabilities
    Return likelihood value
    """
    print("likelihood")
    p=0
    for i in test_X.split(" "):
        if(i in classwise_words_dict[c].keys()):
            p=p+np.log((classwise_words_dict[c][i]+alpha)/classwise_denominators_dict[c])
        else:
            p=p+np.log(alpha/classwise_denominators_dict[c])
    return p

def predict(test_X,classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alpha):
    """
    Complete this function
    """
    print("predict")
    prob=0
    for c in classes:
        p=compute_likelihood(test_X,c,classwise_words_dict,classwise_denominators_dict,alpha)
        p=p+np.log(prior_prob[c])
        if(p>prob or prob==0):
            prob=p
            cl=c
    return cl

def calculate_accuracy(pred_y,test_y,Print=0):
    print("accuracy")
    acc=0
    tp=0
    fp=0
    fn=0
    for i in range(0,len(pred_y),1):
        if(pred_y[i]==test_y[i]):
            acc=acc+1
            if(pred_y[i]==1):
                tp=tp+1
        elif(pred_y[i]==1):
            fp=fp+1
        elif(pred_y[i]==0):
            fn=fn+1
    pre=tp/(tp+fp)
    rec=tp/(tp+fn)
    if(Print==1):
        print("precision = "+str(pre))
        print("recall = "+str(rec))
        print("F1 score = "+str(2*pre*rec/(pre+rec)))
    return acc/len(pred_y)

def train(train_X,train_Y):
    print("train")
    #train=np.genfromtxt(train_file_path,dtype='str',delimiter='\n',skip_header=1)
    
    for i in range(0,train_X.shape[0],1):
        train_X[i]=preprocessing(train_X[i])
    m=train_X.shape[0]
    val_x=train_X[int(m*0.75):]
    val_y=train_Y[int(m*0.75):]
    train_x=train_X[:int(m*0.75)]
    train_y=train_Y[:int(m*0.75)]
    a=[1,3,10,30,100,300,1000,3000]
    classes=np.unique(train_y)
    classwise_words_dict=class_wise_words_frequency_dict(train_x,train_y)
    prior_prob=compute_prior_probabilities(train_y)
    accuracy=0
    for alph in a:
        classwise_denominators_dict=get_class_wise_denominators_likelihood(classwise_words_dict,alph)
        y_pred=[]
        for i in range(0,val_x.shape[0],1):
            y_pred=y_pred+[predict(val_x[i],classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alph)]
        acc=calculate_accuracy(y_pred,val_y)
        if(acc>accuracy):
            accuracy=acc
            alpha=alph
    classwise_words_dict=class_wise_words_frequency_dict(train_X,train_Y)
    prior_prob=compute_prior_probabilities(train_Y)
    classwise_denominators_dict=get_class_wise_denominators_likelihood(classwise_words_dict,alpha)
    return classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alpha

def test(test_X,test_Y,classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alpha):

    print("test")
    #test_X_file_path='.//test.csv'
    # Load Test Data
    #test_X = np.genfromtxt(test_X_file_path,dtype='str', delimiter='\n',skip_header=1)
    #for i in range(0,test_X.shape[0],1):
        #for j in range(0,len(test_X[i]),1):
            #if(test_X[i][j]==','):
                #test_X[i]=test_X[i][j+1:]
                #break
    #for i in range(0,test_X.shape[0],1):
        #test_X[i]=preprocessing(test_X[i])
    y_pred=[]
    for i in range(0,test_X.shape[0],1):
        y_pred=y_pred+[predict(test_X[i],classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alpha)]
    #np.savetxt('predicted_test_Y.csv',y_pred,delimiter=' ')
    acc=calculate_accuracy(y_pred,test_Y,1)
    print("accuracy = "+str(acc))
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.

    You can load the parameters by reading them from a json file which is present in the SubmissionCode.zip
    For Example:
    model = json.load(open("./model_file.json").read())
    """

    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in SubmissionCode.zip and imported properly.
    """

    # Write Outputs to 'predicted_test_Y.csv' file

    #predicted_Y_values = np.array() #ToDo: Update this to contain predicted outputs
    #with open('predicted_test_Y.csv', 'w') as csv_file:
        #writer = csv.writer(csv_file)
        #writer.writerows(predicted_Y_values)


if __name__ == "__main__":
    data=np.genfromtxt('.//train.csv',dtype='str',delimiter='\n',skip_header=1)
    X=[]
    Y=[]
    l=[]
    for i in range(0,data.shape[0],1):
        for j in range(0,len(data[i]),1):
            if(data[i][j]==','):
                data[i]=data[i][j+1:]
                break
    for i in range(0,data.shape[0],1):
        if(data[i][0]=='0' or data[i][0]=='1'):
            continue
        l=l+[i]
    data=np.delete(data,l,axis=0)
    for i in range(0,data.shape[0],1):
        X+=[data[i][2:]]
        Y+=[data[i][0]]
    np.unique(Y,return_counts=True,return_index=True)
    Y=np.array(Y)
    Y=Y.astype('float64')
    X=np.array(X)
    classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alpha=train(X[:int(X.shape[0]*0.75)],Y[:int(Y.shape[0]*0.75)])
    test(X[int(X.shape[0]*0.75):],Y[int(Y.shape[0]*0.75):],classes,classwise_words_dict,classwise_denominators_dict,prior_prob,alpha)