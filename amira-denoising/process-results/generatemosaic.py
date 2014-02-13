#!/usr/bin/env python

import glob
import subprocess

import PIL.Image
import PIL.ImageTk

from Tkinter import *

class App:
    def __init__(self, master):
        self.lblDir = Label(master, text="Directory")
        self.lblDir.grid(row=0, column=0, sticky=W)

        self.strDir = StringVar();
        self.strDir.set("/home/bzflamas/AmmoniteDenoising/datasets/images/output/gkdtrees-denoise-center-141x152x180/3d-direction/processed/")
        self.txtDir = Entry(master, textvariable=self.strDir, width=60)
        self.txtDir.grid(row=0, column=1, columnspan=2)

        self.lblIter = Label(master, text="Iter")
        self.lblIter.grid(row=1, column=0, sticky=W)

        self.txtIter = Entry(master)
        self.txtIter.grid(row=1, column=1, sticky=W)

        self.lblPatch = Label(master, text="Patch")
        self.lblPatch.grid(row=2, column=0, sticky=W)

        self.txtPatch = Entry(master)
        self.txtPatch.grid(row=2, column=1, sticky=W)

        self.lblPCA = Label(master, text="PCA")
        self.lblPCA.grid(row=3, column=0, sticky=W)

        self.txtPCA = Entry(master)
        self.txtPCA.grid(row=3, column=1, sticky=W)

        self.lblStdDev1 = Label(master, text="StdDev1")
        self.lblStdDev1.grid(row=4, column=0, sticky=W)

        self.txtStdDev1 = Entry(master)
        self.txtStdDev1.grid(row=4, column=1, sticky=W)

        self.lblStdDev2 = Label(master, text="StdDev2")
        self.lblStdDev2.grid(row=5, column=0, sticky=W)

        self.txtStdDev2 = Entry(master)
        self.txtStdDev2.grid(row=5, column=1, sticky=W)

        self.btnAction = Button(master, text="Generate", command=self.__generate_mosaic)
        self.btnAction.grid(row=6, column=0)

        self.lblImg = Label(master, text="Test")
        self.lblImg.grid(row=1, column=2, rowspan=6, sticky=W+E+N+S)

    def __generate_mosaic(self):
        process_command = self.__build_process_command()
        if not process_command: return

        print "Generating mosaic..."
        # subprocess.call(process_command.split())

        print "Displaying image..."
        self.__display_image()

    def __build_process_command(self):
        params = {}

        params["dir"] = self.txtDir.get()
        params["iter"] = self.__get_int_field_value(self.txtIter, 3)
        params["patch"] = self.__get_int_field_value(self.txtPatch, 2)
        params["pca"] = self.__get_int_field_value(self.txtPCA, 3)
        params["stddev1"] = self.__get_int_field_value(self.txtStdDev1, 2)
        params["stddev2"] = self.__get_int_field_value(self.txtStdDev2, 2)

        files_str = ""
        files_str += "{dir}/iter-{iter}-patch-{patch}-pca-{pca}"
        files_str += "-stddev-{stddev1}-{stddev2}-set.png"
        files_str = files_str.format(**params)

        files = glob.glob(files_str)

        if files:
            base_command = ""
            base_command += "montage {files}"
            base_command += " -geometry +10+10"
            base_command += " PNG32:{dir}/mosaic.png"
            return base_command.format(files=" ".join(files), dir=params["dir"])

    def __display_image(self):
        image_path = "{}/mosaic.png".format(self.txtDir.get())

        self.image = PIL.Image.open(image_path)
        self.photo = PIL.ImageTk.PhotoImage(self.image)

        self.lblImg.config(image=self.photo)

    def __get_int_field_value(self, field, width):
        if field.get():
            str_format = "{:0" + width + "d}"
            return str_format.format(int(field.get()))
        else:
            return "*"


def main():
    root = Tk()
    app = App(root)

    root.mainloop()


if __name__ == "__main__":
    main()
