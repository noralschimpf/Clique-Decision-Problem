import Utils.dataloader as dl, Utils.viz
import os

def main():
    dict_data = dl.load_dir(os.path('Data/brunson_south-africa'))
    viz.plotnet(dict_data['nda'])

if __name__ == '__main__': main()