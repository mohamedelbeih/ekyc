import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from rdi_face_verification import *


parser = argparse.ArgumentParser()
parser.add_argument("-I", "--Input", help = "Example: Input/", required = True)
parser.add_argument("-O", "--Output", help = "Example: Output/", required = False, default = "output/")
parser.add_argument("-M", "--Model", help = "Example: models/face_verfication/v1.0/model", required = True)
parser.add_argument("-EM", "--Exectution_Mode", help = "sequential or concurrent", required = False, default = "sequential")
parser.add_argument("-VM", "--Verification_Mode", help = "balanced or aggressive or very_aggressive", required = False, default = "balanced")

args = parser.parse_args()

input_path  = args.Input
output_path = args.Output
exectuion_mode = args.Exectution_Mode
model_path = args.Model
verification_mode = args.Verification_Mode


def write_output(result, case):
    os.makedirs(os.path.join(output_path, case),exist_ok=True)

    # save faces images.
    result.face1.convert_color(Image.BGR)
    result.face1.save(os.path.join(output_path, case, "1.png"))

    result.face2.convert_color(Image.BGR)
    result.face2.save(os.path.join(output_path, case, "2.png"))
    
    # save faces coordinates in the original image.
    coords = "face1: (" + str(result.face1_coordinates[0].x) + ", " + str(result.face1_coordinates[0].y) + ")"
    coords += ", (" + str(result.face1_coordinates[1].x) + ", " + str(result.face1_coordinates[1].y) + ")"
    
    coords += "\nface2: (" + str(result.face2_coordinates[0].x) + ", " + str(result.face2_coordinates[0].y) + ")"
    coords += ", (" + str(result.face2_coordinates[1].x) + ", " + str(result.face2_coordinates[1].y) + ")"

    open(os.path.join(output_path, case, "coords.txt"), "w").write(coords)

    # save the metric.
    with open(os.path.join(output_path, case, "metric.txt"), "w") as file:
        data = verification_mode + " Mode\n"
        data += "verified=" + str(result.verified) + "\n"
        data += "confidence=" + str(result.confidence)
        file.write(data)


def run(case_num):
    id_img_path = os.path.join(input_path, case_num, "processed_img.jpg") 
    id_img = Image.construct(id_img_path)

    if id_img.error.code != 0:
        return
    
    face_img_path = os.path.join(input_path, case_num, "captured_img.jpg")
    face_img = Image.construct(face_img_path)

    if face_img.error.code != 0:
        return
    
    if verification_mode == "balanced":
        output = verify_face(id_img.result, face_img.result, model_path, Balanced)
    elif verification_mode == "aggressive":
        output = verify_face(id_img.result, face_img.result, model_path, Aggressive)
    else:
        output = verify_face(id_img.result, face_img.result, model_path, VeryAgressive)

    if output.error.code != 0:
        return
    
    write_output(output.result, case_num)

 
def sequential_execution():
    for dir in os.listdir(input_path):
        run(dir)


def parallel_exectution():
    with ThreadPoolExecutor(max_workers=5) as executor:
        for dir in os.listdir(input_path):
            executor.submit(run, dir)


def main():
    # check the input folder exists
    if not os.path.exists(input_path):
        print("%s doesn't exist." % (input_path))
        exit(1)

    # initialize the model.
    initialize_model(model_path)
    
    # recognize card and its fields.
    start = time.time()
    if exectuion_mode == "sequential" :
        print("**** RUN SEQUENTIAL TEST ****")
        sequential_execution();
    
    else:
        print("**** RUN PARALLEL TEST ****")
        parallel_exectution()
        pass
    
    exectuion_time = time.time() - start
    print("** %s Time = %f secs"% (exectuion_mode, exectuion_time) )


if __name__ == "__main__":
    main()
