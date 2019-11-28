import argparse
import os
from rdkit import Chem

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename',
                        help="the filename to be organized",
                        type=str)
    parser.add_argument('--output_filename',
                        help="output file name",
                        type=str)
    parser.add_argument('--expected_generation',
                        help="number of generations should be made",
                        type=int)
    args = parser.parse_args()

    input_filename = os.path.realpath(os.path.expanduser(args.input_filename))
    output_filename = os.path.realpath(os.path.expanduser(args.output_filename))
    expected_generation = args.expected_generation

    write = []
    generations = []
    generations_wo_scaffold = []
    scaffolds = []
    properties = []
    with open(input_filename) as baseFile:
        for line in baseFile:
            if ":0]" in line:
                _ = "".join(line.split(":0]"))
                line = "".join(_.split("["))
            write.append(line)

            if "CHEMBL" in line:
                id, scaf, prop = line.split(" ")
                scaffolds.append(scaf)

            elif "gen" in line:
                idx, gen, prop, diff = line.split(" ")

                if "." not in gen:
                    prop = float(prop)
                    properties.append(prop)
                    if all(gen != molecule for molecule in generations):
                        generations.append(gen)

    for gen in generations:
        if all(gen != scaf for scaf in scaffolds):
            generations_wo_scaffold.append(gen)

    validness = len(properties)/expected_generation
    uniqueness = len(generations)/len(properties)
    novelty = len(generations_wo_scaffold)/len(generations)
    mean_property = [sum(properties) / len(properties) for i in range(len(properties))]
    mad = list(map(lambda x, y: abs(x - y), properties, mean_property))
    mad = sum(mad) / len(mad)

    write.insert(0, "VALIDNESS: {:.3f}, UNIQUENESS: {:.3f}, NOVELTY: {:.3f}, MAD: {:.3f}\n".format(validness, uniqueness, novelty, mad))
    write.insert(1, "EXPECTED GENERATION: {}\n".format(expected_generation))
    write.insert(2, "TOTAL VALIDNESS COUNT: {}\n".format(len(properties)))
    write.insert(3, "TOTAL UNIQUENESS COUNT: {}\n".format(len(generations)))
    write.insert(4, "TOTAL NOVELTY COUNT: {}\n\n".format(len(generations_wo_scaffold)))

    with open(output_filename, "w") as newFile:
        for line in write:
            newFile.write(line)