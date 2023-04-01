from pathlib import Path
from confz import ConfZ, ConfZFileSource
from pydantic import BaseModel

import pandas as pd


class VerbAtlasConfig(ConfZ):
    data_directory: str

    CONFIG_SOURCES = ConfZFileSource(file="config/verbatlas.json")


class VerbAtlasFrame(BaseModel):
    idx: str
    name: str
    description: str
    synset: str
    example: str
    gloss: str
    type: str
    frame_elements: dict[str, list[str]]


class VerbAtlas:
    def __init__(self) -> None:
        data_path = Path(VerbAtlasConfig().data_directory)
        frame_arg_map = {}
        self.frame_index = {}
        self.name_index = {}
        self.frame_type_index = {}
        with open(data_path / "VA_frame_pas.tsv") as f:
            for line in f.readlines()[1:]:
                fields = line.strip().split("\t")
                frame_id = fields[0]
                args = fields[1:]
                frame_arg_map[frame_id] = {arg: [] for arg in args}

        with open(data_path / "VA_va2sp.tsv") as f:
            for line in f.readlines()[1:]:
                fields = line.strip().split("\t")
                frame_id = fields[0]
                frame_type = fields[1]
                self.frame_type_index[frame_id] = frame_type
                for i in range(2, len(fields), 2):
                    frame_arg_map[frame_id][fields[i]].extend(fields[i + 1].split("|"))

        with open(data_path / "VA_frame_info.tsv") as f:
            for line in f.readlines()[1:]:
                frame, name, description, synset, example, gloss = line.strip().split(
                    "\t"
                )
                frame_inst = VerbAtlasFrame(
                    idx=frame,
                    name=name,
                    description=description,
                    example=example,
                    synset=synset,
                    gloss=gloss,
                    type=self.frame_type_index.get(frame, "C"),
                    frame_elements=frame_arg_map.get(frame, {}),
                )
                self.frame_index[frame] = frame_inst
                self.name_index[name] = frame_inst

    def get_frame(self, frame_idx: str) -> VerbAtlasFrame:
        return self.frame_index[frame_idx]

    def get_frame_by_name(self, name: str) -> VerbAtlasFrame:
        return self.name_index[name]
