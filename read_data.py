from cassis import *
import os
import html
import sys


if len(sys.argv) > 4:
    print("Please use only 3 arguments")
    sys.exit(1)
if sys.argv[1][-4:] == ".xml":
    xml_file = sys.argv[1]
else:
    sys.exit(1)
if sys.argv[2][-1] == "/":
    dir = sys.argv[2]
else:
    sys.exit(1)
if sys.argv[3][-4:] == ".csv":
    csv_file = sys.argv[3]
else:
    sys.exit(1)

def main(xml_file, dir, csv_file):

    with open(xml_file, "rb") as xml:
        typesystem = load_typesystem(xml)

    xmis = os.listdir(dir)
    with open(csv_file,"w") as file:
        file.write("\t".join(["id", "title", "context", "triplets"]) + "\n")
        for xmi in xmis:
            
            line = []
            
            line.append(xmi[5:-4] + "\t")

            with open(f"{dir}/{xmi}", 'r', encoding='UTF-8') as f:
                xmi_str = str(f.read())
                start = xmi_str.index("sofaString=") + 12
                end = xmi_str.index("<cas:View") - 3
                text = html.unescape(xmi_str[start:end])

            line.append(text)

            with open(f"{dir}/{xmi}", 'rb') as f:
                cas = load_cas_from_xmi(f, typesystem=typesystem)

            triplets = []

            for relation in cas.select('webanno.custom.SemanticRelations'):

                dependent = text[relation.get("Dependent").get("begin"):relation.get("Dependent").get("end")]
                governor = text[relation.get("Governor").get("begin"):relation.get("Governor").get("end")]
                relation = relation.get("Relation")

                triplets.append(f"<triplet> {governor} <sub> {dependent} <obj> {relation}")

            line.append(" ".join(triplets))
            file.write("\t".join(line) + "\n")
            

if __name__ == "__main__":
    main(xml_file, dir, csv_file)