
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import threading as thr 
import sys 

class predict_data():
  
    def __init__(self,path5,path6):
        print("Loadingg...")
        self.path5 = path5
        self.path6 = path6
        self.read_dict()
        
        prediction_data = pd.read_excel(
            path5 + "test_data.xlsx", 
            usecols= ['Security_Nm',
                      "Class_Main_Id",
                      'Sub_Class_Id',
                      'Security_Type',
                      'Security_ISIN']
            )
    
        data_to_predict_clean = self.noise_reducer(
            prediction_data.copy()
            ) 
    
        data_to_predict_clean_list =self.listmaker(data_to_predict_clean) 
        #Converts prediction data to a list   
        
        prediction_vector  = self.vector_representation(
            data_to_predict_clean_list
            )  
    
        self.main_prediction = self.predict_main()
        self.sub_prediction  = self.predict_sub()
        
        self.pred = self.final_dataframe()
        print("Predicted data writen to " + path5)
        print(pred)
     
    def read_dict(self):
        self.words_remove_dict = {}
        with open(self.path6 + "dictionary.txt") as f:
            for row in f:
                row = row.replace("\n", "")
                self.words_remove_dict[row] = "" 
        
    def listmaker(self, dataframe):#Transforms df to list
        data_list = dataframe['Security_Nm'].values.T.tolist() #Transforming to list
        return data_list

    def noise_reducer(self, data): 
        """This function cleans the input data to facilitate machine learning"""
        data['Security_Type'].replace(np.nan, '', inplace=True )
        #Replacing all nan values with an empty cell to prevent errors
        data['Security_Nm'] = (
            data['Security_Nm']
            .str
            .cat(data['Security_Type'], sep =' ')
            )
        #Security type is concatenated to the input data 
        empty = pd.DataFrame(0, index=data.index, columns=data.columns) 
        #Creating df with zeros
        empty = empty.replace(0, " ", regex=True) 
        #Replacing zeros with with whitespace chars
        data['Security_Nm'] = data['Security_Nm'].str.lower() 
        #Converting security names to lowercase       
        data['Security_Nm'] = data['Security_Nm'].str.cat(empty['Security_Nm'], sep ='')
        #Concatenting whitespace characters in front
        data['Security_Nm'] = empty['Security_Nm'].str.cat(data['Security_Nm'], sep ='')
        #Concatenting whitespace characters behind
        data['Security_Nm'] = data.Security_Nm.apply(lambda x: x[1:-1].split(' '))   
        data['Security_Nm'] = data['Security_Nm'].apply(self.replacer)
        data['Security_Nm'] = data['Security_Nm'].str.join(" ")       
        data = data.replace('  ', ' ', regex=True) #Removing dict entries to clean data
        data['Security_Nm'] = data['Security_Nm'].str.strip() #Removing extra whitespace around strings
        return data
  
    def replacer(self, words):
        new_list =  []  
        for word in words:
            if word in self.words_remove_dict:
                word = self.words_remove_dict[word]
        new_list.append(word) 
        return new_list

    def vector_representation(self, prediction, variable = "main"): 
        """This functions translates the data to a sparse matrix with the 
        most common characaters"""         
        try:
            count_vect = pickle.load( 
                open(variable + '_count_vectorizer.pkl', "rb" ) 
                ) #Using pickled classifier
            numerical_rep_prediction  = count_vect.transform(prediction) 
            #Feeding the prediction data into the vectorizer             
            return numerical_rep_prediction     
            
        except:
            sys.exit("No saved data - please run script in 'save' mode vect") 
      
    def predict_main(self):      
       try:            
           classifier_main = pickle.load(
           open( 'main_classifer.pkl', "rb" )
           ) #Using pickled classifier
           predict_main = classifier_main.predict(self.prediction_vector)
           #Predicting main classifications
           return predict_main             
       except:
           sys.exit("No saved data - please run script in 'save' mode main")
           
    def predict_sub(self):
     
        prediction_dataframe = pd.DataFrame(
            columns = ['Security_Nm', 'Class_Main_Id']
            
            )
        prediction_dataframe['Security_Nm']   = self.data_to_predict_clean['Security_Nm']
        prediction_dataframe['Class_Main_Id'] = self.main_prediction
        
        prediction_sub_concat = pd.DataFrame(columns = ['prediction_sub'])
        main_class = 10
        for i in range(9):
              """Filtering out main classification"""
              mask_prediction = (prediction_dataframe['Class_Main_Id'] == main_class)
              filtered_prediction = prediction_dataframe.loc[mask_prediction]
              filtered_prediction_list = self.listmaker(filtered_prediction)
              asset = str(main_class)  
              if filtered_prediction.shape == (0, 2): 
              #Returns no values to prevent errors if dataframe is empty
                    pass
        
              else: 
                  vectorized_prediction_list = self.vector_representation(
                      filtered_prediction_list,
                      asset
                      ) 
                  #Vectorizing                
                  try:         
                      classifier_sub = pickle.load( 
                          open( asset +'_sub_classifer.pkl', "rb" ) )
                          #Using pickled classifier
                      predict_sub = classifier_sub.predict(
                          vectorized_prediction_list
                          )#Predicting sub classifications
                      
                      predict_sub_df = pd.DataFrame(
                          columns = ['prediction_sub'],
                          index =filtered_prediction.index
                          ) #Creating sub dataframe
                      predict_sub_df['prediction_sub'] = predict_sub 
                      #Feeding data into dataframe 
                      prediction_sub_concat = prediction_sub_concat.append(
                          predict_sub_df
                          )                           
                      
                  except:
                      sys.exit("No saved data - please run script in 'save' mode")  
              main_class = main_class + 10
        
        prediction_sub_concat = (
            prediction_sub_concat
            .sort_index()
            ) #Sorting inxed in ascending order        
        prediction_sub_concat = (
            prediction_sub_concat[
                np.isfinite(prediction_sub_concat['prediction_sub'])
                ]
            ) #Dropping rows which contain nan values for sub classifications        
        return prediction_sub_concat
      
    def final_dataframe(self):
        final = pd.DataFrame(columns=['Asset_Key','Security_Nm' , 'Main_Class' , 'Sub_Class'])
        final['Security_Nm'] = self.prediction_data['Security_Nm']
        final['Main_Class']  = self.main_prediction 
        final['Sub_Class']   = self.sub_prediction
        return final
       

class train_classifier():
  
      def __init__(self,path1,path2,path3,path6):
            self.no_values = pd.DataFrame({'Security_Nm': ['No values']})   
            self.path1 = path1 #Classified securities
            self.path2 = path2 #Extra training data main class
            self.path3 = path3 #Extra training data sub class
            self.path6 = path6
            self.read_dict()
        
            self.training_data, self.extra_training_main_data, self.extra_training_sub_data = self.training_data_reader()
            print("Training classifier")    
            """Training main classifier"""
            a = thr.Thread(target = self.main_classification)
            a.start()
            
            """Training sub classifier"""
            b = thr.Thread(target = self.sub_classification)
            b.start()
            a.join()
            b.join()
        
    
    def training_data_reader(self):
        training_data = pd.read_excel(self.path1 + "test_data.xlsx", usecols= ['Security_Nm',"Class_Main_Id",'Sub_Class_Id', 'Security_Type', 'Security_ISIN'])
            
        """ self.training_data is the variable assigned to the dataframe 
          containing the data which is used to train the classifier.
          This dataframe should contain columns called 'Security_Nm',
          "Class_Main_Id",'Sub_Class_Id', and 'Security_Type' """
                
        extra_training_main_data = pd.read_excel(self.path2 + "Extra_training_main_data.xlsx", usecols= ['Security_Nm',"Class_Main_Id"])    
            
        """"self.extra_training_data.xlsx contains extra keywords and 
          associated main classificationsto train the classifier EG 
          "REDACTED This dataframe should contain columns called
          'Security_Nm',"Class_Main_Id"' """
                
        extra_training_sub_data = pd.read_excel(self.path3 + "Extra_training_sub_data.xlsx", usecols= ['Security_Nm',"Class_Main_Id", 'Sub_Class_Id'])               
        
        """"self.extra_training_data.xlsx contains extra keywords and
          associated sub classifications to train the classifier - 
          EG "REDACTED"
          This dataframe should contain columns called 'Security_Nm', 
          'Class_Main_Id','Sub_Class_Id'"""
               
        return training_data, extra_training_main_data, extra_training_sub_data
          


    def pickler(self,name, file ):
        with open(name, 'wb') as fid: #Pickle the classifer
            pickle.dump(file, fid)  
      
    def listmaker(self, dataframe):#Transforms df to list
        data_list = dataframe['Security_Nm'].values.T.tolist() #Transforming to list
        return data_list
  
    def noise_reducer(self, data): 
        """This function cleans the input data"""
        data['Security_Type'].replace(np.nan, '', inplace=True ) 
        #Replacing all nan values with an empty cell to prevent errors
        data['Security_Nm'] = (
            data['Security_Nm']
            .str
            .cat(data['Security_Type'], sep =' '))
        #Security type is concatenated to the input data
            
        empty = pd.DataFrame(
            0,
            index=data.index,
            columns=data.columns
            ) #Creating df with zeros
        
        empty = empty.replace(0, " ", regex=True) 
        #Replacing zeros with with whitespace chars
        
        data['Security_Nm'] = (
            data['Security_Nm']
            .str
            .lower()
            ) #Converting security names to lowercase     
            
        data['Security_Nm'] = (
            data['Security_Nm']
            .str
            .cat(empty['Security_Nm'], sep ='')
            ) #Concatenting whitespace characters in front
            
        data['Security_Nm'] = (
            empty['Security_Nm']
            .str
            .cat(data['Security_Nm'], sep ='')
            )#Concatenting whitespace characters behind

        data['Security_Nm'] = (
            data.Security_Nm
            .apply(lambda x: x[1:-1].split(' '))
            )  

        data['Security_Nm'] = data['Security_Nm'].apply(self.replacer)
        data['Security_Nm'] = data['Security_Nm'].str.join(" ")       
        data = data.replace('  ', ' ', regex=True)
        #Removing dict entries to clean data
        data['Security_Nm'] = data['Security_Nm'].str.strip()
        #Removing extra whitespace around strings
        return data
  
    def replacer(self, words):
        new_list =  []  
        for word in words:
            if word in self.words_remove_dict:
                word = self.words_remove_dict[word]
                new_list.append(word) 
       return new_list
  
    def main_classifier(self, training_vector, training_dataframe):    
        classifier_main = RandomForestClassifier(n_estimators = 200)
        #Defining classifier classifier
        classifier_main.fit(
            training_vector,
            training_dataframe['Class_Main_Id']
            ) #Training classifier 
        name =  'main_classifer.pkl'   
        thr.Thread(
            target= self.pickler,
            args = (name,classifier_main)
            ).start()
   
      
    def sub_classifiers(self, training_vector, training_dataframe, asset):       
       
        classifier_sub = RandomForestClassifier(
            n_estimators = 200
            ) #Defining classifier

        classifier_sub.fit(
            training_vector,
            training_dataframe['Sub_Class_Id']
            ) #Training classifier
        name = asset + '_sub_classifer.pkl'   
        thr.Thread(target= self.pickler, args = (
            name,
            classifier_sub)
            ).start()
   
    def main_classification(self):
        classifier_data_appended_main = (
            self.training_data
            .copy()
            .append(self.extra_training_main_data)
            ) #Appending the main training data together with the extra training data
        
        classifier_training_data_appended_main_clean = self.noise_reducer(
            classifier_training_data_appended_main
            ) #Sending the main training data through the data cleaner
        
        classifier_training_data_appended_main_clean_list = self.listmaker(
            classifier_training_data_appended_main_clean
            )  #Converting main training data to a list
            
        #Vectorising and predicting the main data  
        numerical_vector_training_appended_data  = self.vector_representation(
            classifier_training_data_appended_main_clean_list
            )
            
        self.main_classifier(
            numerical_vector_training_appended_data,
            classifier_training_data_appended_main_clean
            )
            
    def sub_classification(self):
        training_data_sub_appended = (
            self.training_data
            .copy()
            .append(self.extra_training_sub_data)
            )          
        training_dataframe = self.noise_reducer(training_data_sub_appended)
        main_class = 10
        for i in range(9):
            """Filtering out main classification"""
            mask_training = (training_dataframe['Class_Main_Id'] == main_class)
            filtered_training = training_dataframe.loc[mask_training]
            filtered_training_list = self.listmaker(filtered_training)
            asset = str(main_class)  
            vectorized_training_list = self.vector_representation(
                filtered_training_list,
                asset) #Vectorizing
                
            self.sub_classifiers(
                vectorized_training_list,
                filtered_training,
                asset
                ) #Training sub classifiers
            main_class = main_class + 10             
    
    def vector_representation(self, training, variable = "main"):
        """This functions translates the data to
            a sparse matrix with the most common characaters """
        
        count_vect = CountVectorizer(analyzer = "char_wb")
        #Creating the vectorizer
        numerical_rep_training = count_vect.fit_transform(training)
        #Feeding the training data into the vectorizer and
        #defining the characters to analyse
        name = variable + '_count_vectorizer.pkl'   
        thr.Thread(target= self.pickler, args = (name,count_vect)).start()
           
        return numerical_rep_training  
    
    def read_dict(self):
        self.words_remove_dict = {}
        with open(self.path6 + "dictionary.txt") as f:
            for row in f:
                row = row.replace("\n", "")
                self.words_remove_dict[row] = ""
        
class console():
    
    def __init__(self,command = None):
        self.path1,self.path2,self.path3,self.path4, self.path5, self.path6 = self.read_paths()
        self.argdict = {
            'help':self.helper,
            'commands':self.commands, 
            'train':self.train, 
            'predict':self.predict, 
            'path':self.paths,
            }

        self.command = command
        self.fetch_commands()
  
    def read_paths(self):
        with open('paths.txt') as f:
            path1 = f.readline().strip()
            path2 = f.readline().strip()
            path3 = f.readline().strip()
            path4 = f.readline().strip()
            path5 = f.readline().strip()
            path6 = f.readline().strip()
        return path1,path2,path3, path4, path5, path6
  
    def fetch_commands(self):
        if self.command[0] in self.argdict:
            self.argdict[self.command[0]]()
      
        else:
            print(self.command[0] + " is not a valid argument, please type 'commands' for a list of commands")

    def commands(self):
        print("\nThe following commands are available:")
        print("\nhelp     - Provides assistance")
        print("commands - Prints out available commands")
        print("train    - Trains the classifier based on data passed to it")
        print("predict  - Predicts the classification of a list of funds")
        print("path     - Changes the path to input/output file ")
  
    def helper(self):
        if len(self.command) == 1:
            print("For help with a specific command, type 'help command'")
            print("For a full list of commands type 'commands'")
        else:
            if self.command[1] == "path":
                print("'path' takes two arguements: type, path_to_file")
                print("\nThere are 7 types:")
                print("1. training_data")
                print("2. extra_training_main_data")
                print("3. extra_training_sub_data")
                print("4. prediction_data")
                print("5. output")
                print("5. dictionary")
                print("6. all")
                print("These are the different input files used to predict data and the output location")
                print("\nExample command to change the location of the prediction data:")
                print("classifier.exe path prediction_data C:/Users/txxxxxxx/folder/subfolder/")
                print("\nTHe following functions are also available:")
                print("'path reset' - Reset all paths to their original locations")
                print("'path list'  - Print out the current paths")  
            elif self.command[1] == "help":
                 print("Really?")
        elif self.command[1] == "train":
            print("This command is used to train the classifier. It requires three xlsx files called: ")
            print("1. 'test_data.xlsx'")
            print("2. 'Extra_training_main_data.xlsx'")
            print("3. 'Extra_training_sub_data.xlsx'")
            print("with the followiing columns:")
            print("'Security_Nm'")
            print("'Security_Key'")
            print("'Class_Main_Id'")
            print("'Sub_Class_Id'")
            print("\nThe current paths are:")
            print("1: " + self.path1)
            print("2: " + self.path2)
            print("3: " + self.path3)
            print("\nTo change the output location, use:")
            print("1: 'path training_data'")
            print("2: 'path extra_training_main_data'")
            print("2: 'path extra_training_sub_data'")
       elif self.command[1] == "predict":
            print("This command is used to classify unclassfied securities. It requires an xlsx file called 'prediction_data.xlsx', with the following columns:")
            print("'Security_Nm'")
            print("'Security_Key'")
            print("\nIt produces an output file at " +self.path5)
            print("\nTo change the output location, use 'path output'")
        
       elif self.command[1] == "commands":
            print("Lists available commands")
       else:
            print("No help available as " + self.command[1] + " is not a valid command")
         
    def train(self):
         train_classifier(self.path1,self.path2,self.path3,self.path6)
  
    def predict(self):
        predict_data(self.path4,self.path6)
 
    def paths(self):
        if len(self.command) == 1:
            print("Please pass an argument, use 'help path' for more information")
        else:
            if self.command[1] == "reset":
                with open("original_paths.txt") as f:
                   self.path1 = f.readline().strip()
                   self.path2 = f.readline().strip()
                   self.path3 = f.readline().strip()
                   self.path4 = f.readline().strip()
                   self.path5 = f.readline().strip()
                   self.path6 = f.readline().strip()
                   self.path_writer()
            elif self.command[1] == "list":
                print("\ntraining_data : " + self.path1)
                print("\nextra_training_main_data : " + self.path2)
                print("\nextra_training_sub_data : " + self.path3)
                print("\nprediction_data : " + self.path4)
                print("noutput : " + self.path5)
            
            elif self.command[1] == "training_data":
                if len(self.command) < 3:
                    print("Please pass a path, use 'help path' for more information")
                else:
                    self.path1 = self.command[2]
                    self.path_writer()
            elif self.command[1] == "extra_training_main_data":
                if len(self.command) < 3:
                  print("Please pass a path, use 'help path' for more information")
                else:
                  self.path2 = self.command[2]
                  self.path_writer()
            elif self.command[1] == "extra_training_sub_data":
                if len(self.command) < 3:
                    print("Please pass a path, use 'help path' for more information")
                else:
                    self.path4 = self.command[2]
                    self.path_writer()
            elif self.command[1] == "prediction_data":
                if len(self.command) < 3:
                    print("Please pass a path, use 'help path' for more information")
                else:
                    self.path5 = self.command[2]
                    self.path_writer()
                    
            elif self.command[1] == "output":
                if len(self.command) < 3:
                  print("Please pass a path, use 'help path' for more information")
                else:
                  self.path5 = self.command[2]
                  self.path_writer()
            elif self.command[1] == "dictionary":
                if len(self.command) < 3:
                    print("Please pass a path, use 'help path' for more information")
                else:
                    self.path6 = self.command[2]
                    self.path_writer()
              
            elif self.command[1] == "all":
                if len(self.command) < 3:
                    print("Please pass a path, use 'help path' for more information")
                else:
                      self.path1 = self.command[2]
                      self.path2 = self.command[2]
                      self.path3 = self.command[2]
                      self.path4 = self.command[2]
                      self.path5 = self.command[2]
                      self.path6 = self.command[2]
                      
                      self.path_writer()
            else:
                print("path "+ self.command[1] + " is not a valid command")
    
  def path_writer(self):
    with open("paths.txt", "w") as f:
      f.write(self.path1 + "\n")
      f.write(self.path2 + "\n")
      f.write(self.path3 + "\n")
      f.write(self.path4 + "\n")
      f.write(self.path5 + "\n")
      f.write(self.path6 + "\n")



    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
      
