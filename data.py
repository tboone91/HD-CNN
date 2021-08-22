# -*- coding: utf-8 -*-
"""
Standford Cars input pipeline

@author: tboonesifuentes
"""
import tensorflow as tf
import pandas as pd
import os
import keras
from keras.datasets import cifar100
from graphviz import Digraph
import shutil 
import glob
import numpy as np

def load_dataset(path,images_path):


    df=pd.read_csv(path)
        
    
    filenames=df['fname'].values
    labels_3=df['class1'].values
    labels_2=df['class2'].values
    labels_1=df['class3'].values
    
    paths=np.repeat(images_path,filenames.size,axis=0)
    
    

    dataset = tf.data.Dataset.from_tensor_slices((paths,filenames,labels_1,labels_2,labels_3))
    dataset = dataset.map(_parse_function,num_parallel_calls=8)
    
    n_elements= len(df)
    
    
    dataset = dataset.shuffle(buffer_size = n_elements)
    
    # 4. batch
    dataset = dataset.batch(n_elements, drop_remainder=False)
  
    # 6. prefetch
    dataset = dataset.prefetch(1)
    
    iterator = iter(dataset)

    X, lab1,lab2,lab3 = iterator.get_next()
    
    X=X.numpy()
    
    lab1=lab1.numpy()
    
    lab2=lab2.numpy()
    
    lab3=lab3.numpy()
    
    return X,lab1,lab2,lab3


def _parse_function(paths,filenames, label_1,label_2,labels_3):
    image_string = tf.io.read_file(paths+'\\'+filenames)
    img = tf.image.decode_jpeg(image_string,channels=3)
    img=tf.image.resize(img,[32,32])
    
    return img, label_1,label_2,labels_3

def taxonomy(num_classes_l0,num_classes_l1,num_classes_l2,lab1,lab2,lab3):
    
    m0 = [[0 for x in range(num_classes_l1)] for y in range(num_classes_l0)]

    for (t, c) in zip(lab1, lab2):
        m0[t][c] = 1

    m1 = [[0 for x in range(num_classes_l2)] for y in range(num_classes_l1)]

    for (t, c) in zip(lab2, lab3):
        m1[t][c] = 1

    
    taxonomy = [m0, m1]
        
    return taxonomy
        
def draw_taxonomy(taxonomy,LABELS):
   """
   This method draws the taxonomy using the graphviz library.
   :return:
   :rtype: Digraph
    """
   u = Digraph('unix', filename='diagram8',node_attr={'color': 'lightblue2', 'style': 'filled'}, strict=True)
   u.attr(size='6,6')
   u.attr(rankdir="LR")

   for i in range(len(taxonomy[0])):
       u.edge('root', LABELS[0][i], LABELS[0][i])

   for l in range(len(taxonomy)):
       for i in range(len(taxonomy[l])):
           for j in range(len(taxonomy[l][i])):
               if taxonomy[l][i][j] == 1:
                   u.edge(LABELS[l][i], LABELS[l + 1][j])

   return u   





class Birds:
    
    LABELS = [['Alaudidae',	'Alcidae',	'Anatidae',	'Bombycillidae',	'Caprimulgidae',	'Cardinalidae',	'Carduelinae',	'Certhiidae',	'Cerylinae',	'Corvidae',	'Cuculidae',	'Diomedeidae',	'Fregatidae',	
               'Gaviidae',	'Halcyoninae',	'Hirundinidae',	'Icteridae',	'Icteriidae',	'Laniidae',	'Laridae',	'Mimidae',	'Motacillidae',	'Neomorphinae',	'Parulidae',	'Passerellidae',	'Passeridae',	
               'Pelecanidae',	'Phalacrocoracidae',	'Picidae',	'Podicipedidae',	'Procellariidae',	'Ptilonorhynchidae',	'Sittidae',	'Stercorariidae',	'Sturnidae',	'Trochilidae',	'Troglodytidae',	
               'Tyrannidae',	'Vireonidae'],
              
              ['Aechmophorus',	'Aethia',	'Agelaius',	'Ailuroedus',	'Ammodramus',	'Ammospiza',	'Amphispiza',	'Anas',	'Anthus',	'Antrostomus',	'Aphelocoma',	'Archilochus',	'Bombycilla',	'Calypte',	
               'Campylorhynchus',	'Cardellina',	'Cardinalis',	'Carduelis',	'Centronyx',	'Cepphus',	'Cerorhinca',	'Certhia',	'Ceryle',	'Chlidonias',	'Chloroceryle',	'Cistothorus',	'Coccyzus',	'Colaptes',	
               'Colibri',	'Contopus',	'Corvus',	'Crotophaga',	'Cyanocitta',	'Cyanocorax',	'Dolichonyx',	'Dryobates',	'Dryocopus',	'Dumetella',	'Empidonax',	'Eremophila',	'Euphagus',	'Fratercula',	
               'Fregata',	'Fulmarus',	'Gavia',	'Geococcyx',	'Geothlypis',	'Haemorhous',	'Halcyon',	'Helmitheros',	'Hesperiphona',	'Hirundo',	'Hydroprogne',	'Icteria',	'Icterus',	'Junco',	'Lamprotornis',	
               'Lanius',	'Larus',	'Leiothlypis',	'Leuconotopicus',	'Leucosticte',	'Limnothlypis',	'Lophodytes',	'Lurocalis',	'Mareca',	'Megaceryle',	'Melanerpes',	'Melospiza',	'Mergus',	'Mimus',	
               'Mniotilta',	'Molothrus',	'Myiarchus',	'Nucifraga',	'Oreoscoptes',	'Pagophila',	'Parkesia',	'Passer',	'Passerculus',	'Passerella',	'Passerina',	'Pelecanus',	'Petrochelidon',	
               'Pheucticus',	'Phoebastria',	'Phoebetria',	'Picoides',	'Pinicola',	'Pipilo',	'Piranga',	'Podiceps',	'Podilymbus',	'Pooecetes',	'Protonotaria',	'Pyrocephalus',	'Quiscalus',	'Riparia',	'Rissa',	
               'Salpinctes',	'Sayornis',	'Seiurus',	'Selasphorus',	'Setophaga',	'Sitta',	'Spinus',	'Spizella',	'Stercorarius',	'Sterna',	'Sternula',	'Sturnella',	'Tachycineta',	'Thalasseus',	'Thryomanes',	
               'Thryothorus',	'Toxostoma',	'Troglodytes',	'Tyrannus',	'Urile',	'Vermivora',	'Vireo',	'Xanthocephalus',	'Zonotrichia'],
               
              ['Acadian_Flycatcher',	'American_Crow',	'American_Goldfinch',	'American_Pipit',	'American_Redstart',	'American_Three_toed_Woodpecker',	'Anna_Hummingbird',	'Artic_Tern',	'Baird_Sparrow',	
               'Baltimore_Oriole',	'Bank_Swallow',	'Barn_Swallow',	'Bay_breasted_Warbler',	'Belted_Kingfisher',	'Bewick_Wren',	'Black_and_white_Warbler',	'Black_billed_Cuckoo',	'Black_capped_Vireo',	'Black_footed_Albatross',	
               'Black_Tern',	'Black_throated_Blue_Warbler',	'Black_throated_Sparrow',	'Blue_Grosbeak',	'Blue_headed_Vireo',	'Blue_Jay',	'Blue_winged_Warbler',	'Boat_tailed_Grackle',	'Bobolink',	'Bohemian_Waxwing',	
               'Brandt_Cormorant',	'Brewer_Blackbird',	'Brewer_Sparrow',	'Bronzed_Cowbird',	'Brown_Creeper',	'Brown_Pelican',	'Brown_Thrasher',	'Cactus_Wren',	'California_Gull',	'Canada_Warbler',	
               'Cape_Glossy_Starling',	'Cape_May_Warbler',	'Cardinal',	'Carolina_Wren',	'Caspian_Tern',	'Cedar_Waxwing',	'Cerulean_Warbler',	'Chestnut_sided_Warbler',	'Chipping_Sparrow',	'Chuck_will_Widow',	'Clark_Nutcracker',	
               'Clay_colored_Sparrow',	'Cliff_Swallow',	'Common_Raven',	'Common_Tern',	'Common_Yellowthroat',	'Crested_Auklet',	'Dark_eyed_Junco',	'Downy_Woodpecker',	'Eared_Grebe',	'Eastern_Towhee',	'Elegant_Tern',	
               'European_Goldfinch',	'Evening_Grosbeak',	'Field_Sparrow',	'Fish_Crow',	'Florida_Jay',	'Forsters_Tern',	'Fox_Sparrow',	'Frigatebird',	'Gadwall',	'Geococcyx',	'Glaucous_winged_Gull',	'Golden_winged_Warbler',	
               'Grasshopper_Sparrow',	'Gray_Catbird',	'Gray_crowned_Rosy_Finch',	'Gray_Kingbird',	'Great_Crested_Flycatcher',	'Great_Grey_Shrike',	'Green_Jay',	'Green_Kingfisher',	'Green_tailed_Towhee',	'Green_Violetear',	
               'Groove_billed_Ani',	'Harris_Sparrow',	'Heermann_Gull',	'Henslow_Sparrow',	'Herring_Gull',	'Hooded_Merganser',	'Hooded_Oriole',	'Hooded_Warbler',	'Horned_Grebe',	'Horned_Lark',	'Horned_Puffin',	'House_Sparrow',	
               'House_Wren',	'Indigo_Bunting',	'Ivory_Gull',	'Kentucky_Warbler',	'Laysan_Albatross',	'Lazuli_Bunting',	'Le_Conte_Sparrow',	'Least_Auklet',	'Least_Flycatcher',	'Least_Tern',	'Lincoln_Sparrow',	'Loggerhead_Shrike',	
               'Long_tailed_Jaeger',	'Louisiana_Waterthrush',	'Magnolia_Warbler',	'Mallard',	'Mangrove_Cuckoo',	'Marsh_Wren',	'Mockingbird',	'Mourning_Warbler',	'Myrtle_Warbler',	'Nashville_Warbler',	'Nelson_Sharp_tailed_Sparrow',	
               'Nighthawk',	'Northern_Flicker',	'Northern_Fulmar',	'Northern_Waterthrush',	'Olive_sided_Flycatcher',	'Orange_crowned_Warbler',	'Orchard_Oriole',	'Ovenbird',	'Pacific_Loon',	'Painted_Bunting',	'Palm_Warbler',	'Parakeet_Auklet',	
               'Pelagic_Cormorant',	'Philadelphia_Vireo',	'Pied_billed_Grebe',	'Pied_Kingfisher',	'Pigeon_Guillemot',	'Pileated_Woodpecker',	'Pine_Grosbeak',	'Pine_Warbler',	'Pomarine_Jaeger',	'Prairie_Warbler',	'Prothonotary_Warbler',	
               'Purple_Finch',	'Red_bellied_Woodpecker',	'Red_breasted_Merganser',	'Red_cockaded_Woodpecker',	'Red_eyed_Vireo',	'Red_faced_Cormorant',	'Red_headed_Woodpecker',	'Red_legged_Kittiwake',	'Red_winged_Blackbird',	'Rhinoceros_Auklet',	
               'Ring_billed_Gull',	'Ringed_Kingfisher',	'Rock_Wren',	'Rose_breasted_Grosbeak',	'Ruby_throated_Hummingbird',	'Rufous_Hummingbird',	'Rusty_Blackbird',	'Sage_Thrasher',	'Savannah_Sparrow',	'Sayornis',	'Scarlet_Tanager',	
               'Scissor_tailed_Flycatcher',	'Scott_Oriole',	'Seaside_Sparrow',	'Shiny_Cowbird',	'Slaty_backed_Gull',	'Song_Sparrow',	'Sooty_Albatross',	'Spotted_Catbird',	'Summer_Tanager',	'Swainson_Warbler',	'Tennessee_Warbler',	
               'Tree_Sparrow',	'Tree_Swallow',	'Tropical_Kingbird',	'Vermilion_Flycatcher',	'Vesper_Sparrow',	'Warbling_Vireo',	'Western_Grebe',	'Western_Gull',	'Western_Meadowlark',	'Western_Wood_Pewee',	'Whip_poor_Will',	
               'White_breasted_Kingfisher',	'White_breasted_Nuthatch',	'White_crowned_Sparrow',	'White_eyed_Vireo',	'White_necked_Raven',	'White_Pelican',	'White_throated_Sparrow',	'Wilson_Warbler',	'Winter_Wren',	'Worm_eating_Warbler',	
               'Yellow_bellied_Flycatcher',	'Yellow_billed_Cuckoo',	'Yellow_breasted_Chat',	'Yellow_headed_Blackbird',	'Yellow_throated_Vireo',	'Yellow_Warbler']]
    
    def __init__(self): 
        
        
        #Download images from url
        
        train_data_url='https://data.deepai.org/CUB200(2011).zip'
        
        filename="CUB200(2011).zip"
        
        untar_filename='CUB_200_2011.tgz'
        
        dataset_path=keras.utils.get_file(filename,train_data_url,extract=True)
        
        dataset_path=dataset_path[0:-len(filename)]
        
        untar_path=dataset_path+untar_filename
        
        sourceDir = dataset_path+'CUB_200_2011\\images\\'
            
        destDir = dataset_path+'CUB_200_2011\\all_images'
        

        if os.path.exists(destDir) is False:
            
            os.mkdir(destDir)
            
            #untar dataset and move images
            shutil.unpack_archive(untar_path,dataset_path)
            
            print('Preparing dataset..')
        
            for jpgfile in glob.iglob(os.path.join(sourceDir, "*", "*.jpg")):
            
                shutil.copy(jpgfile, destDir) 
                
            os.remove(untar_path)
        
        train_csv_url='https://docs.google.com/uc?export=download&id=1sO-Rq64gY96C8vBpzZxoWnPoPixz8-_0'
        train_path=keras.utils.get_file("train.csv", train_csv_url)
    
        
        test_csv_url='https://docs.google.com/uc?export=download&id=1ezqAtXRQju-nsd_pZ9aS6jYmXox4KMh_'
        test_path=keras.utils.get_file("test.csv", test_csv_url)
        
    
        X,lab1,lab2,lab3 = load_dataset(path=train_path,images_path=destDir)
        X_test,lab1t,lab2t,lab3t = load_dataset(path=test_path,images_path=destDir)
        
        self.X_train=X

        self.X_test=X_test[4000:]
        
        self.X_val=X_test[:4000]
        
        self.y_top_test=lab1t
        self.y_c_test=lab2t
        self.y_f_test=lab3t
        
        
        self.image_size = self.X_train[0].shape
        
        self.y_train = [lab1, lab2, lab3]
        
        self.y_val = [lab1t[:4000], lab2t[:4000], lab3t[:4000]]
        
        self.y_test = [lab1t[4000:], lab2t[4000:], lab3t[4000:]]
        
        
        y,idx=tf.unique(lab1)
        self.num_classes_l0 = tf.size(y).numpy()
        y,idx=tf.unique(lab2)
        self.num_classes_l1 = tf.size(y).numpy()
        y,idx=tf.unique(lab3)
        self.num_classes_l2 = tf.size(y).numpy()     
        
        self.taxonomy = taxonomy(self.num_classes_l0,self.num_classes_l1,self.num_classes_l2,lab1,lab2,lab3)
        
        self.draw_taxonomy= draw_taxonomy(self.taxonomy,self.LABELS)


    
    
class Cars:
    
    LABELS = [['Hatchback',	'Sedan','Crew Cab',	'SUV',	'Convertible','Coupe','Wagon','Hatchback Coupe','Van','Minivan','Extended Cab',	'Regular Cab',	'Coupe Convertible'],
              
              ['Coupe Convertible',	'Acura Hatchback',	'Acura Sedan',	'AM General Crew Cab',	'AM General SUV',	'Aston Convertible',	'Aston Coupe',	'Audi Sedan',	'Audi Wagon',	
               'Audi Coupe',	'Audi Convertible',	'Audi Hatchback',	'Bentley Sedan',	'Bentley Coupe',	'Bentley Convertible',	
               'BMW Convertible',	'BMW Coupe',	'BMW Sedan',	'BMW Wagon',	'BMW SUV',	'Bugatti Convertible',	'Bugatti Coupe','Buick SUV',	'Buick Sedan',	'Cadillac Sedan',	
               'Cadillac Crew Cab',	'Cadillac SUV',	'Chevrolet Crew Cab',	'Chevrolet Convertible',	'Chevrolet Coupe',	'Chevrolet Hatchback Coupe',	'Chevrolet Van',	'Chevrolet Minivan',	
               'Chevrolet Sedan',	'Chevrolet Extended Cab',	'Chevrolet Regular Cab',	'Chevrolet SUV',	'Chevrolet Wagon',	'Chrysler Sedan',	'Chrysler SUV',	'Chrysler Convertible',	
               'Chrysler Minivan',	'Daewoo Wagon',	'Dodge Wagon',	'Dodge Minivan',	'Dodge Coupe',	'Dodge Sedan',	'Dodge Extended Cab',	'Dodge Crew Cab',	'Dodge SUV',	'Dodge Van',	
               'Eagle Hatchback',	'Ferrari Convertible',	'Ferrari Coupe',	'FIAT Hatchback',	'FIAT Convertible',	'Fisker Sedan',	'Ford SUV',	'Ford Van',	'Ford Regular Cab',	'Ford Crew Cab',	
               'Ford Sedan',	'Ford Minivan',	'Ford Coupe',	'Ford Convertible','Ford Extended Cab',	'Geo Convertible',	'GMC SUV',	'GMC Extended Cab',	'GMC Van',	'Honda Coupe',	'Honda Sedan',	
               'Honda Minivan',	'Hyundai Sedan',	'Hyundai Hatchback',	'Hyundai SUV',	'Infiniti Coupe',	'Infiniti SUV',	'Isuzu SUV',	'Jaguar Hatchback Coupe',	'Jeep SUV',	'Lamborghini Coupe',	
               'Land Rover SUV',	'Lincoln Sedan',	'Maybach Convertible',	'Mazda SUV',	'McLaren Coupe',	'Mercedes-Benz Convertible',	'Mercedes-Benz Sedan',	'Mercedes-Benz Coupe',	'Mercedes-Benz Van',	
               'MINI Convertible',	'Mitsubishi Sedan',	'Nissan Coupe',	'Nissan Hatchback',	'Nissan Van',	'Plymouth Coupe',	'Porsche Sedan',	'Ram Minivan',	'Rolls-Royce Sedan',	'Rolls-Royce Coupe Convertible',	
               'Scion Hatchback',	'Smart Convertible',	'Spyker Convertible',	'Spyker Coupe',	'Suzuki Sedan',	'Suzuki Hatchback',	'Tesla Sedan',	'Toyota SUV',	'Toyota Sedan',	'Volkswagen Hatchback',	'Volvo Sedan',	
               'Volvo Hatchback',	'Volvo SUV'],
               
              ['Acura Integra Type R 2001',	'Acura RL Sedan 2012',	'Acura TL Sedan 2012',	'Acura TL Type-S 2008',	'Acura TSX Sedan 2012',	'Acura ZDX Hatchback 2012',	'AM General  HUMMER H2 SUT Crew Cab 2009',	
               'AM General HUMMER H3T Crew Cab 2010',	'AM General Hummer SUV 2000',	'Aston Martin V8 Vantage Convertible 2012',	'Aston Martin V8 Vantage Coupe 2012',	'Aston Martin Virage Convertible 2012',	
               'Aston Martin Virage Coupe 2012',	'Audi 100 Sedan 1994',	'Audi 100 Wagon 1994',	'Audi A5 Coupe 2012',	'Audi R8 Coupe 2012',	'Audi RS 4 Convertible 2008',	'Audi S4 Sedan 2007',	'Audi S4 Sedan 2012',	
               'Audi S5 Convertible 2012',	'Audi S5 Coupe 2012',	'Audi S6 Sedan 2011',	'Audi TT Hatchback 2011',	'Audi TT RS Coupe 2012',	'Audi TTS Coupe 2012',	'Audi V8 Sedan 1994',	'Bentley Arnage Sedan 2009',	
               'Bentley Continental Flying Spur Sedan 2007',	'Bentley Continental GT Coupe 2007',	'Bentley Continental GT Coupe 2012',	'Bentley Continental Supersports Conv. Convertible 2012',	
               'Bentley Mulsanne Sedan 2011',	'BMW 1 Series Convertible 2012',	'BMW 1 Series Coupe 2012',	'BMW 3 Series Sedan 2012',	'BMW 3 Series Wagon 2012',	'BMW 6 Series Convertible 2007',	
               'BMW ActiveHybrid 5 Sedan 2012',	'BMW M3 Coupe 2012',	'BMW M5 Sedan 2010',	'BMW M6 Convertible 2010',	'BMW X3 SUV 2012',	'BMW X5 SUV 2007',	'BMW X6 SUV 2012',	'BMW Z4 Convertible 2012',	
               'Bugatti Veyron 16.4 Convertible 2009',	'Bugatti Veyron 16.4 Coupe 2009',	'Buick Enclave SUV 2012',	'Buick Rainier SUV 2007',	'Buick Regal GS 2012',	'Buick Verano Sedan 2012',	
               'Cadillac CTS-V Sedan 2012',	'Cadillac Escalade EXT Crew Cab 2007',	'Cadillac SRX SUV 2012',	'Chevrolet Avalanche Crew Cab 2012',	'Chevrolet Camaro Convertible 2012',	'Chevrolet Cobalt SS 2010',	
               'Chevrolet Corvette Convertible 2012',	'Chevrolet Corvette Ron Fellows Edition Z06 2007',	'Chevrolet Corvette ZR1 2012',	'Chevrolet Express Cargo Van 2007',	'Chevrolet Express Van 2007',	'Chevrolet HHR SS 2010',
               'Chevrolet Impala Sedan 2007',	'Chevrolet Malibu Hybrid Sedan 2010',	'Chevrolet Malibu Sedan 2007',	'Chevrolet Monte Carlo Coupe 2007',	'Chevrolet Silverado 1500 Classic Extended Cab 2007',	
               'Chevrolet Silverado 1500 Extended Cab 2012',	'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',	'Chevrolet Silverado 1500 Regular Cab 2012',	'Chevrolet Silverado 2500HD Regular Cab 2012',	
               'Chevrolet Sonic Sedan 2012',	'Chevrolet Tahoe Hybrid SUV 2012',	'Chevrolet TrailBlazer SS 2009',	'Chevrolet Traverse SUV 2012',	'Chrysler 300 SRT-8 2010',	'Chrysler Aspen SUV 2009',	
               'Chrysler Crossfire Convertible 2008',	'Chrysler PT Cruiser Convertible 2008',	'Chrysler Sebring Convertible 2010',	'Chrysler Town and Country Minivan 2012',	'Daewoo Nubira Wagon 2002',	'Dodge Caliber Wagon 2007',	
               'Dodge Caliber Wagon 2012',	'Dodge Caravan Minivan 1997',	'Dodge Challenger SRT8 2011',	'Dodge Charger Sedan 2012',	'Dodge Charger SRT-8 2009',	'Dodge Dakota Club Cab 2007',	'Dodge Dakota Crew Cab 2010',	
               'Dodge Durango SUV 2007',	'Dodge Durango SUV 2012',	'Dodge Journey SUV 2012',	'Dodge Magnum Wagon 2008',	'Dodge Ram Pickup 3500 Crew Cab 2010',	'Dodge Ram Pickup 3500 Quad Cab 2009',	'Dodge Sprinter Cargo Van 2009',	
               'Eagle Talon Hatchback 1998',	'Ferrari 458 Italia Convertible 2012',	'Ferrari 458 Italia Coupe 2012',	'Ferrari California Convertible 2012',	'Ferrari FF Coupe 2012',	'FIAT 500 Abarth 2012',	'FIAT 500 Convertible 2012',	
               'Fisker Karma Sedan 2012',	'Ford Edge SUV 2012',	'Ford E-Series Wagon Van 2012',	'Ford Expedition EL SUV 2009',	'Ford F-150 Regular Cab 2007',	'Ford F-150 Regular Cab 2012',	'Ford F-450 Super Duty Crew Cab 2012',	'Ford Fiesta Sedan 2012',	
               'Ford Focus Sedan 2007',	'Ford Freestar Minivan 2007',	'Ford GT Coupe 2006',	'Ford Mustang Convertible 2007',	'Ford Ranger SuperCab 2011',	'Geo Metro Convertible 1993',	'GMC Acadia SUV 2012',	'GMC Canyon Extended Cab 2012',	
               'GMC Savana Van 2012',	'GMC Terrain SUV 2012',	'GMC Yukon Hybrid SUV 2012',	'Honda Accord Coupe 2012',	'Honda Accord Sedan 2012',	'Honda Odyssey Minivan 2007',	'Honda Odyssey Minivan 2012',	'Hyundai Accent Sedan 2012',	
               'Hyundai Azera Sedan 2012',	'Hyundai Elantra Sedan 2007',	'Hyundai Elantra Touring Hatchback 2012',	'Hyundai Genesis Sedan 2012',	'Hyundai Santa Fe SUV 2012',	'Hyundai Sonata Hybrid Sedan 2012',	'Hyundai Sonata Sedan 2012',	
               'Hyundai Tucson SUV 2012',	'Hyundai Veloster Hatchback 2012',	'Hyundai Veracruz SUV 2012',	'Infiniti G Coupe IPL 2012',	'Infiniti QX56 SUV 2011',	'Isuzu Ascender SUV 2008',	'Jaguar XK XKR 2012',	'Jeep Compass SUV 2012',	
               'Jeep Grand Cherokee SUV 2012',	'Jeep Liberty SUV 2012',	'Jeep Patriot SUV 2012',	'Jeep Wrangler SUV 2012',	'Lamborghini Aventador Coupe 2012',	'Lamborghini Diablo Coupe 2001',	'Lamborghini Gallardo LP 570-4 Superleggera 2012',	
               'Lamborghini Reventon Coupe 2008',	'Land Rover LR2 SUV 2012',	'Land Rover Range Rover SUV 2012',	'Lincoln Town Car Sedan 2011',	'Maybach Landaulet Convertible 2012',	'Mazda Tribute SUV 2011',	'McLaren MP4-12C Coupe 2012',	
               'Mercedes-Benz 300-Class Convertible 1993',	'Mercedes-Benz C-Class Sedan 2012',	'Mercedes-Benz E-Class Sedan 2012',	'Mercedes-Benz S-Class Sedan 2012',	'Mercedes-Benz SL-Class Coupe 2009',	'Mercedes-Benz Sprinter Van 2012',	
               'MINI Cooper Roadster Convertible 2012',	'Mitsubishi Lancer Sedan 2012',	'Nissan 240SX Coupe 1998',	'Nissan Juke Hatchback 2012',	'Nissan Leaf Hatchback 2012',	'Nissan NV Passenger Van 2012',	'Plymouth Neon Coupe 1999',	
               'Porsche Panamera Sedan 2012',	'Ram C/V Cargo Van Minivan 2012',	'Rolls-Royce Ghost Sedan 2012',	'Rolls-Royce Phantom Drophead Coupe Convertible 2012',	'Rolls-Royce Phantom Sedan 2012',	'Scion xD Hatchback 2012',	
               'smart fortwo Convertible 2012',	'Spyker C8 Convertible 2009',	'Spyker C8 Coupe 2009',	'Suzuki Aerio Sedan 2007',	'Suzuki Kizashi Sedan 2012',	'Suzuki SX4 Hatchback 2012',	'Suzuki SX4 Sedan 2012',	
               'Tesla Model S Sedan 2012',	'Toyota 4Runner SUV 2012',	'Toyota Camry Sedan 2012',	'Toyota Corolla Sedan 2012',	'Toyota Sequoia SUV 2012',	'Volkswagen Beetle Hatchback 2012',	'Volkswagen Golf Hatchback 1991',	
               'Volkswagen Golf Hatchback 2012',	'Volvo 240 Sedan 1993',	'Volvo C30 Hatchback 2012',	'Volvo XC90 SUV 2007']]
    
    def __init__(self): 
        
        
        train_data_url='http://ai.stanford.edu/~jkrause/car196/car_ims.tgz'
        
        filename='car_ims.tgz'
        
        print('Preparing dataset..')
        
        dataset_path=keras.utils.get_file(filename,train_data_url,extract=True,untar=True)
        

        train_csv_url='https://docs.google.com/uc?export=download&id=1BPR6qoSr3o1J670NsS_cST31p2Ai3N54'
        train_path=keras.utils.get_file("train_cars.csv", train_csv_url)
    
        
        test_csv_url='https://docs.google.com/uc?export=download&id=1enfdXAi7w93iRz2xDRsu21OOfTxngwCv'
        test_path=keras.utils.get_file("test_cars.csv", test_csv_url)
        
        destDir=dataset_path[0:-len(filename)]+'car_ims'
    
        X,lab1,lab2,lab3 = load_dataset(path=train_path,images_path=destDir)
        X_test,lab1t,lab2t,lab3t = load_dataset(path=test_path,images_path=destDir)
    
        
        self.X_train=X

        self.y_train = [lab1, lab2, lab3]
        
        self.y_val = [lab1t[:2500], lab2t[:2500], lab3t[:2500]]
        
        self.y_test = [lab1t[2500:], lab2t[2500:], lab3t[2500:]]
        
        self.image_size = self.X_train[0].shape
        
        y,idx=tf.unique(lab1)
        self.num_classes_l0 = tf.size(y).numpy()
        y,idx=tf.unique(lab2)
        self.num_classes_l1 = tf.size(y).numpy()
        y,idx=tf.unique(lab3)
        self.num_classes_l2 = tf.size(y).numpy()        

    
        self.taxonomy = taxonomy(self.num_classes_l0,self.num_classes_l1,self.num_classes_l2,lab1,lab2,lab3)
        
        self.draw_taxonomy= draw_taxonomy(self.taxonomy,self.LABELS)
           

mapping_coarse_to_top = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 0, 12: 0, 13: 0, 14: 0,
                         15: 0, 16: 0, 17: 0, 18: 1, 19: 1}


def map_fine_to_cluster_cifar100(y, mapping):
    """
    This function is only used to create label for clusters if used.  Clusters are obtained from:
    :param y:
    :type y:
    :return:
    :rtype:
    """
    # Mapping fine -> cluster

    y_top = []
    for f in y:
        k = f[0]
        c = np.array([mapping[k]])
        y_top.append(c)
    return np.array(y_top)


class Cifar100:
    LABELS = [['bio organism', 'objects'],
              ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetales',
               'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
               'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herivores',
               'medium_mammals', 'non-insect_inverterates', 'people', 'reptiles', 'small_mammals', 'trees',
               'vehicles_1', 'vehicles_2'],
              ['apple', 'aquarium_fish', 'ray', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'owl', 'boy',
               'ridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
               'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
               'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyoard', 'lamp', 'lawn_mower',
               'leopard', 'lion', 'lizard', 'loster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
               'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rait', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
               'streetcar', 'sunflower', 'sweet_pepper', 'tale', 'tank', 'telephone', 'television', 'tiger', 'tractor',
               'train', 'trout', 'tulip', 'turtle', 'wardroe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']]

    def __init__(self):
        """
        :param type: to indicate if to use coarse classes as given in the cifar100 dataset or use clusters.
        :type type: str
        """
        (X_c_train, y_c_train), (X_c_test, y_c_test) = cifar100.load_data(label_mode='coarse')
        (X_f_train, y_f_train), (X_f_test, y_f_test) = cifar100.load_data(label_mode='fine')

        y_top_train = map_fine_to_cluster_cifar100(y_c_train, mapping_coarse_to_top)
        y_top_test = map_fine_to_cluster_cifar100(y_c_test, mapping_coarse_to_top)

        self.X_train = X_f_train
        self.X_val = X_f_test[:5000]
        self.X_test = X_f_test[5000:]

        self.y_train = [y_top_train, y_c_train, y_f_train]
        self.y_val = [y_top_test[:5000], y_c_test[:5000], y_f_test[:5000]]
        self.y_test = [y_top_test[5000:], y_c_test[5000:], y_f_test[5000:]]

        self.image_size = self.X_train[0].shape

        self.num_classes_l0 = len(set([v[0] for v in y_top_train]))
        self.num_classes_l1 = len(set([v[0] for v in y_c_train]))
        self.num_classes_l2 = len(set([v[0] for v in y_f_train]))
        
        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        for (t, c) in zip(y_top_train, y_c_train):
            t = t[0]
            c = c[0]
            m0[t][c] = 1

        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(y_c_train, y_f_train):
            t = t[0]
            c = c[0]
            m1[t][c] = 1
            
        self.taxonomy = [m0, m1]

        self.draw_taxonomy= draw_taxonomy(self.taxonomy,self.LABELS)





if __name__ == '__main__':
    dataset = Cifar100()
    print(dataset.num_classes_l0)
    print(dataset.num_classes_l1)
    print(dataset.num_classes_l2)
    print(dataset.taxonomy)
    u = dataset.draw_taxonomy
    u.view()
    
    


