train = './train.txt'
entity2id = './entity2id.txt'
relation2id = './relation2id.txt'

entity = set()
relation = set()
def write_dict(name):
	if name == "entity":
		read_path = entity2id
		write_path = "./entities.dict"
	else:
		read_path = relation2id
		write_path = "./relations.dict"
	kk = open(write_path, "w")
	with open(read_path, "r") as f:
		for line in f.readlines():
			item, idx = line.strip().split("\t")
			kk.write(idx+"\t"+item+"\n")


write_dict("entity")
write_dict("relation")




