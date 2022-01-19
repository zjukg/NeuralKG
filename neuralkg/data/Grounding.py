from .DataPreprocess import KGData
import pdb
class GroundAllRules:
    def __init__(self, args):
        self.MapRelation2ID = {}
        self.MapEntity2ID = {}
        self.Relation2Tuple = {}
        self.MapID2Entity = {}
        self.MapID2Relation = {}
        self.TrainTriples = {}

        self.RelSub2Obj = {}
        self.MapVariable = {}

        self.args = args
        self.fnEntityIDMap = args.data_path + "/entities.dict"
        self.fnRelationIDMap = args.data_path + "/relations.dict"
        path_len = len(args.data_path.split('/'))
        self.fnRuleType = args.data_path + "/" + args.data_path.split('/')[path_len - 1] + "_rule"
        self.fnTrainingTriples = args.data_path + "/train.txt"
        self.fnOutout = args.data_path + "/groudings.txt"

    def PropositionalizeRule(self):
        self.kgData = KGData(self.args)
        self.readData(self.fnEntityIDMap, self.fnRelationIDMap, self.fnTrainingTriples)
        self.groundRule(self.fnRuleType, self.fnOutout)

    def readData(self, fnEntityIDMap, fnRelationIDMap, fnTrainingTriples):
        tokens = []
        self.MapEntity2ID = self.kgData.ent2id
        self.MapRelation2ID = self.kgData.rel2id
        self.TrainTriples = self.kgData.TrainTriples
        self.Relation2Tuple = self.kgData.Relation2Tuple
        self.RelSub2Obj = self.kgData.RelSub2Obj
        # with open(fnEntityIDMap, 'r', encoding='utf-8') as f:
        #     for line in f.readlines():
        #         line = line.strip('\n')
        #         tokens = line.split("\t")
        #         iEntityID = int(tokens[0])
        #         strValue = tokens[1]
        #         self.MapEntity2ID[strValue] = iEntityID
        #         self.MapID2Entity[iEntityID] = strValue
        #
        # with open(fnRelationIDMap, "r", encoding='utf-8') as f:
        #     for line in f.readlines():
        #         line = line.strip('\n')
        #         tokens = line.split("\t")
        #         iRelationID = int(tokens[0])
        #         strValue = tokens[1]
        #         self.MapRelation2ID[strValue] = iRelationID
        #         self.MapID2Relation[iRelationID] = strValue

        print("Start to load soft rules......")

        # with open(fnTrainingTriples, "r", encoding='utf-8') as f:
        #     for line in f.readlines():
        #         line = line.strip('\n')
        #         tokens = line.split("\t")
        #         iRelationID = self.MapRelation2ID[tokens[1]]
        #         strValue = tokens[0] + "#" + tokens[2]
        #         line = line.replace(tokens[0], str(self.MapEntity2ID[tokens[0]]))
        #         line = line.replace(tokens[1], str(self.MapRelation2ID[tokens[1]]))
        #         line = line.replace(tokens[2], str(self.MapEntity2ID[tokens[2]]))
        #         self.TrainTriples[line] = True
        #         if not iRelationID in self.Relation2Tuple:
        #             tmpLst = []
        #             tmpLst.append(strValue)
        #             self.Relation2Tuple[iRelationID] = tmpLst
        #         else:
        #             self.Relation2Tuple[iRelationID].append(strValue)

        # with open(fnTrainingTriples, "r", encoding='utf-8') as f:
        #     for line in f.readlines():
        #         line = line.strip('\n')
        #         tokens = line.split("\t")
        #         iRelationID = self.MapRelation2ID[tokens[1]]
        #         iSubjectID = self.MapEntity2ID[tokens[0]]
        #         iObjectID = self.MapEntity2ID[tokens[2]]
        #         tmpMap = {}
        #         tmpMap_in = {}
        #         if not iRelationID in self.RelSub2Obj:
        #             if not iSubjectID in tmpMap:
        #                 tmpMap_in.clear()
        #                 tmpMap_in[iObjectID] = True
        #                 tmpMap[iSubjectID] = tmpMap_in
        #             else:
        #                 tmpMap[iSubjectID][iObjectID] = True
        #             self.RelSub2Obj[iRelationID] = tmpMap
        #         else:
        #             tmpMap = self.RelSub2Obj[iRelationID]
        #             if not iSubjectID in tmpMap:
        #                 tmpMap_in.clear()
        #                 tmpMap_in[iObjectID] = True
        #                 tmpMap[iSubjectID] = tmpMap_in
        #             else:
        #                 tmpMap[iSubjectID][iObjectID] = True
        #             self.RelSub2Obj[iRelationID] = tmpMap  # 是不是应该要加？
        print("success")

    def groundRule(self, fnRuleType, fnOutput):
        print("Start to propositionalize soft rules......")
        writer = open(fnOutput, "w")
        tmpLst = {}
        with open(fnRuleType, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("?"):
                    bodys = line.split("=>")[0].strip().split("  ")
                    heads = line.split("=>")[1].strip().split("  ")
                if len(bodys) == 3:
                    bEntity1 = bodys[0]
                    iFstRelation = self.MapRelation2ID[bodys[1]]
                    bEntity2 = bodys[2]
                    bEntity3 = heads[0]
                    iSndRelation = self.MapRelation2ID[heads[1]]
                    bEntity4 = heads[2].split("\t")[0]
                    hEntity1 = heads[2].split("\t")[1]
                    confi = float(hEntity1)
                    iSize = len(self.Relation2Tuple[iFstRelation])

                    for i in range(0, iSize):
                        strValue = self.Relation2Tuple.get(iFstRelation)[i]
                        iSubjectID = self.MapEntity2ID[strValue.split("#")[0]]
                        iObjectID = self.MapEntity2ID[strValue.split("#")[1]]
                        self.MapVariable[bEntity1] = iSubjectID
                        self.MapVariable[bEntity2] = iObjectID
                        strKey = "(" + str(iSubjectID) + "\t" + str(iFstRelation) + "\t" + str(
                            iObjectID) + ")\t" + "(" + str(self.MapVariable[bEntity3]) + "\t" + str(
                            iSndRelation) + "\t" + str(self.MapVariable[bEntity4]) + ")"
                        strCons = str(self.MapVariable[bEntity3]) + "\t" + str(iSndRelation) + "\t" + str(
                            self.MapVariable[bEntity4])
                        if (not strKey in tmpLst) and (not strCons in self.TrainTriples):
                            writer.write("2\t" + str(strKey) + "\t" + str(confi) + "\n")
                            tmpLst[strKey] = True
                        writer.flush()
                        self.MapVariable.clear()
                if len(bodys) == 6:
                    bEntity1 = bodys[0].strip()
                    iFstRelation = self.MapRelation2ID[bodys[1].strip()]
                    bEntity2 = bodys[2].strip()
                    bEntity3 = bodys[3].strip()
                    iSndRelation = self.MapRelation2ID[bodys[4].strip()]
                    bEntity4 = bodys[5].strip()
                    hEntity1 = heads[0].strip()
                    iTrdRelation = self.MapRelation2ID[heads[1].strip()]
                    hEntity2 = heads[2].split("\t")[0].strip()
                    confidence = heads[2].split("\t")[1].strip()
                    confi = float(confidence)
                    mapFstRel = self.RelSub2Obj[iFstRelation]
                    mapSndRel = self.RelSub2Obj[iSndRelation]
                    for lstEntity1 in mapFstRel:
                        self.MapVariable[bEntity1] = lstEntity1
                        lstEntity2 = list(mapFstRel[lstEntity1].keys())
                        iFstSize = len(lstEntity2)

                        for iFstIndex in range(0, iFstSize):
                            iEntity2ID = lstEntity2[iFstIndex]
                            self.MapVariable[bEntity1] = lstEntity1
                            self.MapVariable[bEntity2] = iEntity2ID
                            lstEntity3 = []
                            if (bEntity3 in self.MapVariable) and (self.MapVariable[bEntity3] in mapSndRel):
                                lstEntity3.append(self.MapVariable[bEntity3])
                            else:
                                if not bEntity3 in self.MapVariable:
                                    lstEntity3 = list(mapSndRel.keys())

                            iSndSize = len(lstEntity3)

                            for iSndIndex in range(0, iSndSize):
                                iEntity3ID = lstEntity3[iSndIndex]
                                self.MapVariable[bEntity1] = lstEntity1
                                self.MapVariable[bEntity2] = iEntity2ID
                                self.MapVariable[bEntity3] = iEntity3ID
                                lstEntity4 = []
                                if (bEntity4 in self.MapVariable) and (
                                        self.MapVariable[bEntity4] in mapSndRel[iEntity3ID]):
                                    lstEntity4.append(self.MapVariable[bEntity4])
                                else:
                                    if not bEntity4 in self.MapVariable:
                                        lstEntity4 = list(mapSndRel[iEntity3ID].keys())

                                iTrdSize = len(lstEntity4)

                                for iTrdIndex in range(0, iTrdSize):
                                    iEntity4ID = lstEntity4[iTrdIndex]
                                    self.MapVariable[bEntity4] = iEntity4ID
                                    infer = str(self.MapVariable[hEntity1]) + "\t" + str(iTrdRelation) + "\t" + str(
                                        self.MapVariable[hEntity2])
                                    strKey = "(" + str(lstEntity1) + "\t" + str(iFstRelation) + "\t" + str(
                                        iEntity2ID) + ")\t(" + str(iEntity3ID) + "\t" + str(iSndRelation) + "\t" + str(
                                        iEntity4ID) + ")\t" + "(" + str(self.MapVariable[hEntity1]) + "\t" + str(
                                        iTrdRelation) + "\t" + str(self.MapVariable[hEntity2]) + ")"
                                    if (not strKey in tmpLst) and (not infer in self.TrainTriples):
                                        writer.write("3\t" + strKey + "\t" + str(confi) + "\n")
                                        tmpLst[strKey] = True
                                self.MapVariable.clear()
                            self.MapVariable.clear()
                        writer.flush()
                        self.MapVariable.clear()
