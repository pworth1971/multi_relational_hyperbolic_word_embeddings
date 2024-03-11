import spacy
from typing import Iterable
from tqdm import tqdm
from dynaconf import settings
from saf import Sentence
from .abstract import Annotator


class SpacyAnnotator(Annotator):
    def __init__(self, annot_model = None):
        if not annot_model:
            if (Annotator.default_model):
                self.annot_model = Annotator.default_model
            else:
                self.annot_model = spacy.load(settings["spacy_annotator_model"])
                #Annotator.default_model = self.annot_model

    def annotate(self, definitions: Iterable[Sentence]):
        for defn in tqdm(definitions, desc="Annotating (Spacy)"):
            annots = self.annot_model(defn.surface)
            for i in range(len(defn.tokens)):
                defn.tokens[i].annotations["pos"] = annots[i].pos_
                defn.tokens[i].annotations["lemma"] = annots[i].lemma_
                defn.tokens[i].annotations["dep"] = annots[i].dep_
                defn.tokens[i].annotations["ctag"] = annots[i].tag_
