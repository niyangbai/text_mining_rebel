from cassis import *
import os
import html


xml_file = "TypeSystem.xml"
dir = "./startupminingtobi"
csv_file = 'csvfile.csv'

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