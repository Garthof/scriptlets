#!/usr/bin/env python

import glob
import subprocess

import PIL.Image
import PIL.ImageTk

from Tkinter import *

class App:
    def __init__(self, master):
        # Upper frame
        self.frmDir = Frame(master)
        self.frmDir.pack(fill=X)

        self.lblDir = Label(self.frmDir, text="Directory")
        self.lblDir.pack(side=LEFT)

        self.strDir = StringVar()
        self.strDir.set("/home/bzflamas/AmmoniteDenoising/datasets/images/output/gkdtrees-denoise-center-141x152x180/3d-direction/processed/")
        self.txtDir = Entry(self.frmDir, textvariable=self.strDir)
        self.txtDir.pack(side=LEFT, fill=X, expand=1)

        # Lower frame
        self.frmMain = Frame(master)
        self.frmMain.pack(fill=BOTH, expand=1)

        self.frmFields = Frame(self.frmMain)
        self.frmFields.pack(side=LEFT)

        self.frmIter = Frame(self.frmFields)
        self.frmIter.pack()
        self.lblIter = Label(self.frmIter, text="Iter", width=10)
        self.lblIter.pack(side=LEFT)
        self.strIter = StringVar()
        self.strIter.set("1")
        self.txtIter = Entry(self.frmIter, textvariable=self.strIter)
        self.txtIter.pack(side=LEFT)

        self.frmPatch = Frame(self.frmFields)
        self.frmPatch.pack()
        self.lblPatch = Label(self.frmPatch, text="Patch", width=10)
        self.lblPatch.pack(side=LEFT)
        self.strPatch = StringVar()
        self.strPatch.set("5")
        self.txtPatch = Entry(self.frmPatch, textvariable=self.strPatch)
        self.txtPatch.pack(side=LEFT)

        self.frmPCA = Frame(self.frmFields)
        self.frmPCA.pack()
        self.lblPCA = Label(self.frmPCA, text="PCA", width=10)
        self.lblPCA.pack(side=LEFT)
        self.strPCA = StringVar()
        self.strPCA.set("3")
        self.txtPCA = Entry(self.frmPCA, textvariable=self.strPCA)
        self.txtPCA.pack(side=LEFT)

        self.frmStdDev1 = Frame(self.frmFields)
        self.frmStdDev1.pack()
        self.lblStdDev1 = Label(self.frmStdDev1, text="StdDev1", width=10)
        self.lblStdDev1.pack(side=LEFT)
        self.txtStdDev1 = Entry(self.frmStdDev1)
        self.txtStdDev1.pack(side=LEFT)

        self.frmStdDev2 = Frame(self.frmFields)
        self.frmStdDev2.pack()
        self.lblStdDev2 = Label(self.frmStdDev2, text="StdDev2", width=10)
        self.lblStdDev2.pack(side=LEFT)
        self.txtStdDev2 = Entry(self.frmStdDev2)
        self.txtStdDev2.pack(side=LEFT)

        self.btnAction = Button(self.frmFields, text="Generate", command=self.__generate_mosaic)
        self.btnAction.pack()

        self.canvas = Canvas(self.frmMain)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=1)
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))

    def __generate_mosaic(self):
        process_command = self.__build_process_command()
        if not process_command:
            print "No images to show"

        print "Generating mosaic..."
        subprocess.call(process_command.split())

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
            base_command += " -tile 4x"
            base_command += " -geometry +10+10"
            base_command += " PNG32:{dir}/mosaic.png"
            return base_command.format(files=" ".join(files), dir=params["dir"])

    def __display_image(self):
        image_path = "{}/mosaic.png".format(self.txtDir.get())

        self.image = PIL.Image.open(image_path)
        self.image = self.image.resize((self.image.size[0]/2, self.image.size[1]/2))
        self.photo = PIL.ImageTk.PhotoImage(self.image)
        self.photo_id = self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def __get_int_field_value(self, field, width):
        if field.get():
            str_format = "{:0" + str(width) + "d}"
            return str_format.format(int(field.get()))
        else:
            return "*"


def main():
    root = Tk()
    app = App(root)

    root.mainloop()


if __name__ == "__main__":
    main()
