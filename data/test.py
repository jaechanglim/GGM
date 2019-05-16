if __name__ == "__main__":
    with open("./ChEMBL+STOCK1S/data_ChEMBL_normalized.txt", "w") as newFile:
        with open("./ChEMBL/data_normalized.txt") as oldFile:
            for line in oldFile:
                newFile.write(line)
        with open("./affinity/data_normalized.txt") as oldFile:
            for line in oldFile:
                data = line.split("\t")
                molId = data[0]
                newLine = molId + " None None\n"
                newFile.write(newLine)