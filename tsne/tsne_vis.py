# coding: utf-8

from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class TSNE_Visualizer():
    """ t-SNE による特徴量埋め込み表示のヘルパーモジュール

    特徴表現ベクトル、クラスラベル、文字列を入力として、画像埋め込み、
    ラベルテキスト埋め込み、Scatterのプロット出力を行う。
    t-SNEによる次元圧縮は `scikit-learn.manifold.TSNE` を使用する。
    グラフ出力は `matplotlib.pyplot` , `seaborn` を使用する。
    グラフ出力はt-SNEの埋め込み後次元数が2の場合に対応。

    t-SNE処理の都合上、データ数(data_num)は10000以内程度を推奨。

    埋め込み後の値の範囲、ラベル、文字列を使用してフィルタ可能。

    Args:
        features (ndarray):
            特徴ベクトルのリスト。[data_num, vactor]の2次元配列。
        
        labels (list of data):
            特徴ベクトルに対応するクラスラベルのリスト。[data_num] の配列。
            textplotのラベル、scatterのhueとして使用する。
        
        paths (list of pathlike):
            特徴ベクトルに対応する文字列のリスト。[data_num] の配列。
            image_plot で読み込む画像のパスとして使用する。
        
        n_components (int):
            t-SNEによる埋め込み後の次元数。3次元で埋め込みを行うことも可能
            だが、グラフ描画には対応していない。
        
        do_exec (bool):
            インスタンス生成時に `fit_transform()` を実行するかどうか。
            実行しない場合は手動で `fit_transform()` を呼び出すこと。

        tsne_model (ndarray):
            すでにt-SNEによる特徴ベクトル表現に変換したものを指定する場合は
            このパラメータで指定する。do_exec が Falseのときのみメンバ変数を更新する。
            filter後の特徴量を利用して本オブジェクトを生成する場合に利用する。

        **kwargs (dict):
            `scikit-learn.manifold.TSNE()` に渡すkwargs。
    """
    def __init__(self, features, labels, paths=None, n_components=2, do_exec=False, tsne_model=False, **kwargs):
        self.features = features
        self.labels = labels
        self.tsne = TSNE(n_components=n_components, **kwargs)
        self.paths = paths
        if do_exec:
            self.fit_transform()
        elif tsne_model is not None:
            self.tsne_model = tsne_model

    def fit_transform(self):
        """ t-SNEの埋め込みを実行する。
        """
        self.tsne_model = self.tsne.fit_transform(self.features)

    def plot_image_map(self, canvas_size, tile_size, figsize, title=None, outpath=None):
        """ image map を生成する。

        t-SNEで次元削減した2次元座標に、pathsで指定した画像を表示する。
        cavas_size, tile_size, figsize からグリッドのサイズと数が決まる。
        複数の特徴量が同一のグリッドに位置する場合、後に読み込んだ画像に置き換える。

        Args:
            canvas_size (tuple):
                image map のサイズ(pixel)。(xsize, ysize)の要素数2のタプル。
            tile_size (tuple):
                グリッドに張り付ける画像のサイズ(pixel)。(xsize, ysize)の要素数2のタプル。
            figsize (tuple):
                axesのサイズ(inch)。(xsize, ysize)の要素数2のタプル。
            title (str):
                image mapのタイトル。
            outpath (path-like):
                生成した image map の保存先パス。指定しなかった場合は画像を保存しない。
        Returns:
            AxesImage。生成した image map のAxes。
        """
        if self.paths == None:
            raise Exception('plot_image_map need paths of Images, please specify.')
        canvas = Image.new('RGB', canvas_size)
        val_max = np.array(self.tsne_model).max()
        val_min = np.array(self.tsne_model).min()

        for i, path in enumerate(self.paths):
            x, y = self.tsne_model[i]
            pos_x = int(x*(canvas_size[0]/tile_size[0])/(val_max-val_min))*tile_size[0]
            pos_y = int(y*(canvas_size[1]/tile_size[1])/(val_max-val_min))*tile_size[1]
            pos = (int(pos_x+canvas_size[0]/2), int(canvas_size[1]-(pos_y+canvas_size[1]/2)))
            target_img = Image.open(path)
            target_img = target_img.resize(tile_size)
            canvas.paste(target_img, pos)
            target_img.close()

        plt.figure(figsize=figsize)
        ret = plt.imshow(np.array(canvas).transpose(1,0,2))
        if title:
            ret = plt.title(title)
        if outpath:
            ret = plt.savefig(outpath)
        return ret

    def plot_text_map(self, labels, figsize, title=None, outpath=None):
        """ text map を生成する。

        t-SNEで生成した特徴量でtext mapを生成する。散布図でラベルをクラス番号にしたもの。

        Args:
            labels (list of data or None):
                データに対応したクラスラベルのリスト。 `self.tsne_model` と同じ長さにする必要がある。
                指定しなかった場合はメンバ変数のlabelsを使用する。
            figsize (tuple):
                axesのサイズ(inch)。(xsize, ysize)の要素数2のタプル。
            title (str):
                text map のタイトル。
            outpath (path-like):
                text map の保存先。指定しなかった場合は画像を保存しない。
        Returns:
            AxesImage。生成した text map のAxes。
        """
        if not labels:
            labels = self.labels
        xmin = self.tsne_model[:, 0].min()
        xmax = self.tsne_model[:, 0].max()
        ymin = self.tsne_model[:, 1].min()
        ymax = self.tsne_model[:, 1].max()

        plt.figure(figsize=figsize)
        for _y, _label in zip(self.tsne_model, labels):
            plt.text(_y[0], _y[1], _label)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel("component 0")
        ret = plt.ylabel("component 1")
        if title:
            ret = plt.title(title)
        if outpath:
            ret = plt.savefig(outpath)
        return ret

    def plot_scatter(self, labels=None, figsize=(16, 16), title=None, outpath=None):
        """ scatter(散布図) を生成する。

        t-SNEで生成した特徴量でscatterを生成する。クラスはhueとして使用する。

        Args:
            labels (list of int):
                データに対応したクラスラベルのリスト。 `self.tsne_model` と同じ長さにする必要がある。
                指定しなかった場合はメンバ変数のlabelsを使用する。
            figsize (tuple):
                axesのサイズ(inch)。(xsize, ysize)の要素数2のタプル。
            title (str):
                text map のタイトル。
            outpath (path-like):
                text map の保存先。指定しなかった場合は画像を保存しない。
        Returns:
            AxesImage。生成した scatter のAxes。
        """
        if not labels:
            labels = self.labels
        plt.figure(figsize=figsize)
        ret = sns.scatterplot(self.tsne_model[:,0], self.tsne_model[:,1], labels)
        if title:
            ret = plt.title(title)
        if outpath:
            ret = plt.savefig(outpath)
        return ret

    def filter(self, f_tsne_feat=None, f_labels=None, f_paths=None, update=False):
        """ 内部のデータに対してフィルタを適用し、データを返します。

        features, labels, paths のうち、指定した条件を満たすデータだけを抽出し
        そのデータを返します。update=Trueの場合、インスタンスのメンバ変数をフィルタ適用後の
        データに更新します。
        複数種別のフィルタはすべてandで処理します。

        Args:
            f_tsne_feat (ndarray):
                t-SNE適用後の特徴量に適用するフィルタ。適用後の次元数分のminとmaxを指定すること。
                2次元の場合 `numpy.array([[xmin, xmax], [ymin, ymax]])` 。
                各境界値は含むデータのみにフィルタする。
            f_labels (list of int):
                抽出したいクラスラベルのリスト。指定したクラスラベルを持つデータのみにフィルタする。
            f_paths (list of path-like):
                抽出したい文字列を含むリスト。指定した文字列に部分一致するデータのみにフィルタする。
            update (bool):
                インスタンスのメンバ変数をフィルタ後のデータに置き換えるかどうか。
        Returns:
            _f_tsne_model (ndarray):
                フィルタ適用後のt-SNE特徴ベクトルのリスト。
            _labels (list of int):
                フィルタ適用後のクラスラベルのリスト。
            _paths (list of path-like):
                フィルタ適用後のファイルパスのリスト。
        """
        data_num = len(self.tsne_model)
        conds = np.array([True] * data_num)
        if f_tsne_feat is not None:
            if f_tsne_feat.shape[0] != self.tsne_model.shape[1]:
                raise TypeError('f_features.shape[0]:{} is not match self.tsne_model.shape[1]:{}'.format(f_tsne_feat.shape[0], self.tsne_model.shape[1]))

            # self.features から、それぞれの次元でminとmax の範囲内にあるデータを抽出する
            for i in range(f_tsne_feat.shape[0]):
                conds &= self.tsne_model[:, i] >= f_tsne_feat[i][0]
                conds &= self.tsne_model[:, i] <= f_tsne_feat[i][1]

        if f_labels:
            # labelsのリストのうち、値がf_labelsのものだけにフィルタリングする
            _c = np.array([False] * data_num)
            for fl in f_labels:
                _c |= np.array(self.labels) == fl
            conds &= _c

        if f_paths:
            # pathsのリストのうち、特定の文字列を含むものだけにフィルタリングする
            _c = np.array([False] * data_num)
            for f_path in f_paths:
                __c = []
                for p in self.paths:
                    __c.append(True) if f_path in p else __c.append(False)
                _c |= np.array(__c)
            conds &= _c

        indexes = np.where(conds)
        _tsne_model = self.tsne_model[indexes]
        _labels = list(np.array(self.labels)[indexes[0]])
        _paths = list(np.array(self.paths)[indexes[0]])

        if update:
            self.tsne_model = _tsne_model
            self.labels = _labels
            self.paths = _paths

        return _tsne_model, _labels, _paths

def make_infer_gt_labels(y_trues, y_preds):
    """ 
    """