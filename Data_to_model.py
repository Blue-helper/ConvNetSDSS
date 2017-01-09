import AtlasToFits as atf
import image_creator as ic
import Downsampler as ds

def Data_to_model(todownload, contador, aux_list, Objid):
    atf.atlastofits(todownload, contador, aux_list, Objid)
    ic.image_creator()
    ds.downsampler()
