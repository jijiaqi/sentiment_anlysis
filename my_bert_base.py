import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import os
from transformers import *
from transformers import BertTokenizer

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    """默认返回input_ids,token_type_ids,attention_mask"""
    inputs = tokenizer.encode_plus(instance,
        add_special_tokens=True,
        max_length=max_sequence_length,
        truncation_strategy= 'longest_first')

    input_ids =  inputs["input_ids"]
    input_masks = inputs["attention_mask"]
    input_segments = inputs["token_type_ids"]
    padding_length = max_sequence_length - len(input_ids)
    #填充
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)
    return [input_ids, input_masks, input_segments]    
    

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
        input_ids, input_masks, input_segments = [], [], []
        for instance in tqdm(df[columns]):
        
            ids, masks, segments = _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)
        
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)

        return [np.asarray(input_ids, dtype=np.int32), 
                np.asarray(input_masks, dtype=np.int32), 
                np.asarray(input_segments, dtype=np.int32)
               ]
    
def compute_output_arrays(df, columns):
        return np.asarray(df[columns].astype(int) + 1)    
    
class modelBERT(tf.keras.Model):
    def __init__(self):
        super(modelBERT, self).__init__(name='first_bert')
        config = BertConfig.from_pretrained('https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json',output_hidden_states=True) 
        self.bert_model = TFBertModel.from_pretrained('/data/jjq/bert-base-chinese-tf_model.h5', config=config)
        self.concat =  tf.keras.layers.Concatenate(axis=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.15)
        self.output_ = tf.keras.layers.Dense(3, activation='softmax')
   
        
    def call(self, inputs):
        input_id,input_mask,input_atn = inputs
        sequence_output, pooler_output, hidden_states  = self.bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)
        h12 = tf.reshape(hidden_states[-1][:,0],(-1,1,768))
        h11 = tf.reshape(hidden_states[-2][:,0],(-1,1,768))
        h10 = tf.reshape(hidden_states[-3][:,0],(-1,1,768))
        h09 = tf.reshape(hidden_states[-4][:,0],(-1,1,768)) 
        concat_hidden = self.concat(([h12, h11, h10, h09]))
        x = self.avgpool(concat_hidden)
        x = self.dropout(x)
        x = self.output_(x)
        
        return x
        
        
    
           

if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 140
    input_categories = '微博中文内容'
    output_categories = '情感倾向'
    
    #清洗读数据
    df_train = pd.read_csv('nCoV_100k_train.labled.csv',engine ='python')
    df_train = df_train[df_train[output_categories].isin(['-1','0','1'])]
    df_test = pd.read_csv('nCov_10k_test.csv',engine ='python')
    df_sub = pd.read_csv('submit_example.csv')
    print('train shape =', df_train.shape)
    print('test shape =', df_test.shape)
    
    #得到bert模型需要的ids,mask,segment结构
    tokenizer = BertTokenizer.from_pretrained('https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt')
    inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    outputs = compute_output_arrays(df_train, output_categories)
    
    #添加损失函数
    def Focal_Loss(y_true, y_pred, alpha=0.5, gamma=2):
        """
        focal loss for multi-class classification
        fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
        :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
        :param y_pred: prediction after softmax shape of [batch_size, nb_class]
        :param alpha:
        :param gamma:
        :return:
        """
        y_pred += tf.keras.backend.epsilon()
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        fl = ce * weight * alpha
        reduce_fl = tf.keras.backend.max(fl, axis=-1)
        return reduce_fl
    
    
    #模型的切分以及训练
    gkf = StratifiedKFold(n_splits=5).split(X=df_train[input_categories].fillna('-1'),y=df_train[output_categories].fillna('-1'))

    valid_preds = []
    test_preds = []
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = to_categorical(outputs[train_idx])

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = to_categorical(outputs[valid_idx])

        model = modelBERT()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        train_dataset=tf.data.Dataset.from_tensor_slices(((train_inputs[0],train_inputs[1],train_inputs[2]),train_outputs)).shuffle(buffer_size=1000).batch(32)
        
       
        

                                                                   
        
        valid_dataset= tf.data.Dataset.from_tensor_slices(((valid_inputs[0],valid_inputs[1],valid_inputs[2]),valid_outputs)).batch(32)
        
                  
        
    
        FL=lambda y_true,y_pred: Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2)
    
        model.compile(loss=FL, optimizer=optimizer, metrics=['acc'])   
        model.fit(train_inputs, train_outputs, validation_data= [valid_inputs, valid_outputs], epochs=2, batch_size=32)
        valid_preds.append(model.predict(valid_inputs))
        test_preds.append(model.predict(test_inputs))
        K.clear_session()
        
        
    #将数据转化为提交的csv格式
    
    sub = np.average(test_preds, axis=0)
    sub = np.argmax(sub,axis=1)
    df_sub['y'] = sub-1
    df_sub.columns=['id','y']
    df_sub.to_csv('test_sub.csv',index=False, encoding='utf-8')
    

'''

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    k=tokenizer.encode_plus("人民大学",
        add_special_tokens=True,
        max_length=20,
        truncation_strategy= 'longest_first')
       print(k)
'''
    