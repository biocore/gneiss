import qiime.plugin.model as model

from gneiss.regression import RegressionModel


class RegressionFormat(model.BinaryFileFormat):
    def sniff(self):
        try:
            res = RegressionModel.read_pickle(open(str(self)))
            return isinstance(res, RegressionModel)
        except:
            return False


RegressionDirectoryFormat = model.SingleFileDirectoryFormat(
    'RegressionDirectoryFormat', 'regression.pickle', RegressionFormat)
