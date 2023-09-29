from cm_engine import *
import os
import argparse
import time
import sys
import math
from concurrent.futures import ThreadPoolExecutor

import threading

#PS: You can add inference threads , use double inference , uniquify values as arguments  and pass them to detect card function instaed of using default values like we do now 
parser = argparse.ArgumentParser()
parser.add_argument("-I", "--Input", help = "Example: Input/", required = True)
parser.add_argument("-O", "--Output", help = "Example: Output/", required = False, default = "output/")
parser.add_argument("-RM", "--RecognitionModel", help = "Example: RecognitionModel/", required = True)
parser.add_argument("-PM", "--PreprocessorModel", help = "Example: PreprocessorModel/", required = True)
parser.add_argument("-EM", "--Exectution_Mode", help = "sequential or concurrent", required = False, default = "sequential")
parser.add_argument("-MC", "--MultiCard", help = "pass 1 if input image has multiple cards", required = False, default = 0)

args = parser.parse_args()

input_path  = args.Input
output_path = args.Output
exectuion_mode = args.Exectution_Mode
recogn_models_path = args.RecognitionModel
preprocessor_model_path = args.PreprocessorModel
multicard = int(args.MultiCard)
inference_threads = 8

# create full directory path if not exist.
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_text_fields(fields, output_dir_path, file_name):
    # format output.
    content = ""
    for field in fields:
        content += field.type + ":" + field.value + "\n"
    
    # write it.
    create_dir(output_dir_path)
    file_path = output_dir_path + "/" + file_name
    
    with open(file_path, 'w', encoding="utf-8") as writer:
        writer.write(content)

def dump_image_fields(fields, output_dir_path):
    create_dir(output_dir_path)
    for field in fields:
        field.image.save(output_dir_path + "/" + field.type + ".png")
    

def print_error(msg, error):
    print("\033[1;31m-- %s => Error(%s, %i)\033[0m" % (msg, error.what, error.code))

def name_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

    
def get_card_corners(axes):
    corners = PointVector()
    for i in range(0, len(axes), 2):
        corners.push_back(Point(int(axes[i]), int(axes[i+1])))
        
    return corners
    
def read_card_corners_n_type(corners_path):
    items = (open(corners_path, "r").readlines()[0]).split()
    
    card_type = items[-1]
    card_corners = get_card_corners(items[:-1])
    
    return card_corners, card_type
    
def read_image(path):
    output = Image.construct(path, Image.BGR)
    
    if output.error.code != 0:
        print_error("Failed to read image : %s" % path, output.error)
        return None
    
    return output


def recognize_single_card(image_path, output_path, card_type, card_corners):
    # read the image.  
    image_output = read_image(image_path)
    if image_output is None: return
    
    #calculate card image blurriness percentage.
    blur_result = get_blur_percentage(image_output.result, preprocessor_model_path)
    if blur_result.error.code != 0:
        return print_error("Failed to caculate blurrines percentage of image : %s" % image_path, blur_result.error)
   
    print("-- %s has blurriness percentage = %f" % (image_path, blur_result.result))
        
    output = recognize_card(image_output.result, preprocessor_model_path, recogn_models_path, inference_threads, card_type, card_corners, True ,8)
    recognized_card, error = output.result, output.error
    
    if error.code != 0:
        return print_error("Failed to Recognize Card of image : %s" % image_path, error)
    
    recognized_text_fields  = recognized_card.text_fields
    recognized_image_fields = recognized_card.image_fields

    # dump the card fields.
    write_text_fields(recognized_text_fields, output_path, name_without_ext(image_path) + ".txt")
    if len(recognized_image_fields) > 0:
        print('------------- dump image fields of ' + name_without_ext(image_path) + ' -------------- ')
        dump_image_fields(recognized_image_fields, output_path + "/" + name_without_ext(image_path))


def recognize_multi_cards(image_path, output_path):
    # read the image.  
    image_output = read_image(image_path)
    if image_output is None: return
    
    # detect the card and its fields.
    output = detect_multi_cards(image_output.result, preprocessor_model_path, inference_threads)
    cards, error = output.result, output.error
    
    if error.code != 0:
        return print_error("Failed to Detect Card of image : %s" % image_path, error)
    
    # recognize the card's fields.
    for i in range(cards.size()):
        card_output = recognize_text_fields(cards[i].fields, recogn_models_path)

        if card_output.error.code != 0:
            print_error("Failed to Recognize fields of Card in image : %s" % image_path, card_output.error )
            continue

        # dump the card fields.
        write_text_fields(card_output.result, output_path+ "/" + name_without_ext(image_path), str(i) + ".txt")
        

def run(image_path,):  
    image_name = name_without_ext(image_path)
      
    if multicard == 1:
        print("-- MultiCard Recognition -> ", image_name)
        recognize_multi_cards(image_path, output_path + "/multiCard/")
    else:
        print("-- SingleCard Recognition [Default]-> ", image_name)
        recognize_single_card(image_path, output_path + "/default/", "Non", PointVector())

        # run custom card if the card side and corners are given.
        corners_path = os.path.join(input_path, image_name+".txt")
        if  os.path.exists(corners_path):
            print("-- SingleCard Recognition [Custom]-> ", image_name)
            card_corners, card_type = read_card_corners_n_type(corners_path)
            recognize_single_card(image_path, output_path + "/custom/", card_type, card_corners)
 
 
def sequential_execution():
    for image_name in os.listdir(input_path):
        if image_name.endswith(".txt"): continue
        run(input_path + "/" + image_name)


def parallel_exectution():
    with ThreadPoolExecutor(max_workers=5) as executor:
        for image_name in os.listdir(input_path):
            if image_name.endswith(".txt"): continue
            executor.submit(run, input_path + "/" + image_name)
               
def main():
    # check the input folder exists
    if not os.path.exists(input_path):
        print("%s doesn't exist." % (input_path))
        exit(1)

    # initialize the model.
    initialize_model(preprocessor_model_path, recogn_models_path)
    
    # recognize card and its fields.
    start = time.time()
    if exectuion_mode == "sequential" :
        print("**** RUN SEQUENTIAL TEST ****")
        sequential_execution()
    
    else:
        print("**** RUN PARALLEL TEST ****")
        parallel_exectution()
        pass
    
    exectuion_time = time.time() - start;
    print("** %s Time = %f secs"% (exectuion_mode, exectuion_time) )

    pass

if __name__ == "__main__":
    main()

