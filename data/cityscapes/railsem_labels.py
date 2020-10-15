from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id   color
    Label(  'buffer-stop'          ,  0 , ( 70, 70, 70) ),
    Label(  'crossing'             ,  1 , (128, 64,128) ),
    Label(  'guard-rail'           ,  2 , (  0,255,  0) ),
    Label(  'train-car'            ,  3 , (100, 80,  0) ),
    Label(  'platform'             ,  4 , (232, 35,244) ),
    Label(  'rail'                 ,  5 , (255,255,  0) ),
    Label(  'switch-indicator'     ,  6 , (127,255,  0) ),
    Label(  'switch-left'          ,  7 , (255,255,  0) ),
    Label(  'switch-right'         ,  8 , (127,127,  0) ),
    Label(  'switch-unknown'       ,  9 , (191,191,  0) ),
    Label(  'switch-static'        , 10 , (  0,255,127) ),
    Label(  'track-sign-front'     , 11 , (  0,220,220) ),
    Label(  'track-signal-front'   , 12 , ( 30,170,250) ),
    Label(  'track-signal-back'    , 13 , (  0, 85,125) ),
    #rail occluders
    Label(  'person-group'         , 14 , ( 60, 20,220) ),
    Label(  'car'                  , 15 , (142,  0,  0) ),
    Label(  'fence'                , 16 , (153,153,190) ),
    Label(  'person'               , 17 , ( 60, 20,220) ),
    Label(  'pole'                 , 18 , (153,153,153) ),
    Label(  'rail-occluder'        , 19 , (255,255,255) ),
    Label(  'truck'                , 20 , ( 70,  0,  0) ),
]

def get_train_ids():
    train_ids = []
    for i in labels:
        train_ids.append(i.id)
    return train_ids