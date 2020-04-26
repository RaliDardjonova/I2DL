import numpy as np 


def transform_data(data, values_key): 
    import math
    def isNaN(num):
        return num != num
    def one_hot_embedding(labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.

        Returns:
        (tensor) encoded labels, sized [N, #classes].
        """
        y = np.identity(num_classes) 
        return y[labels] 


        
    # split data into numeric and categoric

    data_num = data.select_dtypes(include=['float', 'int'])
    data_cat = data.select_dtypes(include='object')

    #replace values in dict by numeric 
    for key, sub_dict in values_key.items():
    
        for sub_key, value in sub_dict.items():
            if isNaN(sub_key):
                data_cat.loc[data_cat[key].isna() , key] = value
            else:
                data_cat.loc[data_cat[key] == sub_key, key] = value

    # transform data to torch tensor
    numpy_num = data_num.values
    numpy_cat = data_cat.values



    def StandardizeData( data):
        """Scales data to mean 0 and std 0

        Args:
        data

        Returns:
        (tensor) Normalised data
        """
        
        return (data - np.mean( data, axis= 0))/np.std( data, axis = 0)
        
        

    print(np.sum(np.isnan(numpy_num) ))
    print(np.sum(np.isnan(numpy_cat) ))

    nr_data, num_features = np.shape(numpy_cat)
    print( nr_data, num_features)

    one_hot_data = [] 
    for index,feat in enumerate( data_cat.columns):
        one_hot_data.append(one_hot_embedding( numpy_cat[:, index], len(values_key[feat])) )
    numpy_one_hot =   np.concatenate( one_hot_data, axis= 1)
    numpy_num_standard = StandardizeData( numpy_num)


    all_data = np.concatenate(( numpy_num_standard , numpy_one_hot) , axis= 1)


    # fill torch num nan with -1000

    all_data[np.isnan( all_data)] = -1000

    print(all_data)
    return all_data

