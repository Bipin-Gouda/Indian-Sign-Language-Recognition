# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense , Dropout,GlobalAveragePooling2D
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications import VGG19#,preprocess_input    #from keras.applications.vgg19 import VGG19,preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #@@@ 1 tha
sz = 128     # @@@@@@@ 128 tha    What is 128 here? ie our img size is 128*128 pixels

print("Enter the Algorithm number to run")
print("1.CNN (1 conv layers) 2.CNN (2 conv layers) 3.CNN (3 conv layers) 4.ANN 5.VGG19 6.InceptionV3 7.Resnet50")
temp='6'
print("algorithm number selected is {}".format(temp))


if temp=='1':       # CNN with 1 convolution layer
    cnn1 = Sequential()
    cnn1.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(sz,sz,1)))
    cnn1.add(MaxPooling2D(pool_size=(2, 2)))
    cnn1.add(Dropout(0.2))
    
    cnn1.add(Flatten())

    cnn1.add(Dense(128, activation='relu'))
    cnn1.add(Dense(10, activation='softmax'))

    cnn1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if temp=='2':    # CNN with 2 convolution layers
    # Step 1 - Building the CNN
    
    # Initializing the CNN
    classifier = Sequential()
    
    # First convolution layer and pooling
    classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Second convolution layer and pooling
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    #classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening the layers       ..................  FLATTENING HERE
    classifier.add(Flatten())
    
    # Adding a fully connected layer
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.40))
    classifier.add(Dense(units=96, activation='relu'))    # WE have 4 dense layers
    classifier.add(Dropout(0.40))
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=29, activation='softmax')) # softmax for more than 2        # 27 tha
    
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

if temp == '3':
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(sz,sz,1)))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Convolution2D(128, (3, 3), activation='relu'))
    classifier.add(Dropout(0.4))

    classifier.add(Flatten())

    classifier.add(Dense(128, activation='relu'))     # WE have 2 Dense ,layerz
    classifier.add(Dropout(0.3))
    classifier.add(Dense(29, activation='softmax'))

    classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


if temp=='5':
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)  #512
    x = Dropout(0.3)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # train the model on the new data for a few epochs
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])    #'categorical_accuracy'])
    
    model.summary()
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('data/train',
                                                     target_size=(sz, sz),
                                                     batch_size=16,               #10
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory('data/test',                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=16,        #10
                                                class_mode='categorical')
    start=time.time()
    
    history=model.fit(                                        
            training_set,
            steps_per_epoch=680, # No of images in training set  #12841 tha ()   #680
            epochs=5,           #5 tha   #10
            validation_data=test_set,
            validation_steps=140)# No of images in test set #4268 tha  #140
    
    end=time.time()
    print("Time elasped is : ",end-start,"seconds")
    
    

if temp=='6':
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('data/train',
                                                     target_size=(sz, sz),
                                                     batch_size=16,               #10
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory('data/test',                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=16,        #10
                                                class_mode='categorical') 
    
    # train the model on the new data for a few epochs
    

    
    history=model.fit(                                        
            training_set,
            steps_per_epoch=680, # No of images in training set  #12841 tha ()   #680
            epochs=10,           #5 tha   #10
            validation_data=test_set,
            validation_steps=140)# No of images in test set #4268 tha  #140
    


if temp=='7':
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np

    model = ResNet50(weights='imagenet')

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(29, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('data/train',
                                                     target_size=(sz, sz),
                                                     batch_size=16,               #10
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory('data/test',                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=16,        #10
                                                class_mode='categorical') 
    
    # train the model on the new data for a few epochs
    
    history=model.fit(                                        
            training_set,
            steps_per_epoch=680, # No of images in training set  #12841 tha ()   #680
            epochs=10,           #5 tha   #10
            validation_data=test_set,
            validation_steps=140)# No of images in test set #4268 tha  #140
   

if temp=='1' or temp=='2' or temp=='3':
    # Step 2 - Preparing the train/test data and training the model
    
    classifier.summary()
    
    
    # Code copied from - https://keras.io/preprocessing/image/
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('data/train',
                                                     target_size=(sz, sz),
                                                     batch_size=16,               #10
                                                     color_mode='grayscale',
                                                     class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory('data/test',                        # data/test
                                                target_size=(sz , sz),   
                                                batch_size=16,        #10
                                                color_mode='grayscale',
                                                class_mode='categorical') 
    history=classifier.fit(                                         # added history= lateron
            training_set,
            steps_per_epoch=680, # No of images in training set  #12841 tha ()   #680
            epochs=10,           #5 tha   #10
            validation_data=test_set,
            validation_steps=140)# No of images in test set #4268 tha  #140
    
    ################### YEplot bad me dala h maine
    
    # 1. Accuracy And Val_accuracy plot
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1])
    plt.legend(loc='lower right')
    
    #2.  Loss and Val_loss plot 
    
''' plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0,5])
    plt.ylim([0.015,3])
    plt.legend(loc='lower right')
    #test_loss, test_acc = classifier.evaluate(test_images,  test_labels, verbose=2) plot k sath ka mal h'''

    
if temp=='4':           #Building the ANN

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,      # max value can be 255 so scaling everything bw 0 to 1 
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    training_set = train_datagen.flow_from_directory('data/train',
                                                     target_size=(sz, sz),
                                                     batch_size=10,   #16        #10
                                                     color_mode='grayscale',
                                                     class_mode='categorical',)
    
    ann = Sequential([
        Flatten(input_shape=(sz,sz,1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(29, activation='softmax')    
    ])

    ann.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    history=ann.fit(training_set, epochs=3,steps_per_epoch=17)
    
    '''plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([0,3])
    plt.ylim([0.015, 1])
    plt.legend(loc='lower right')'''
    
    
    
    
# Saving the model
   
if temp=='1':   # CNN model to be saved ANN just for display
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\model")
    print(os.getcwd())
    model_json = classifier.to_json()
    with open("model-bw.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    classifier.save_weights('model-bw.h5')
    print('Weights saved')
    
if temp=='3':   # CNN Conv 3
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN3conv")
    print(os.getcwd())
    model_json = classifier.to_json()
    with open("model-bw3.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    classifier.save_weights('model-bw3.h5')
    print('Weights saved')
    
if temp=='5':   
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\VGG19conv")
    print(os.getcwd())
    model_json = model.to_json()
    with open("model-bwVGG19.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    model.save_weights('model-bwVGG19.h5')
    print('Weights saved')

if temp=='6':   
    os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\InceptionV3conv")
    print(os.getcwd())
    model_json = model.to_json()
    with open("model-bwInceptionV3.json", "w") as json_file:
        json_file.write(model_json)
    print('Model Saved')
    model.save_weights('model-bwInceptionV3.h5')
    print('Weights saved')
