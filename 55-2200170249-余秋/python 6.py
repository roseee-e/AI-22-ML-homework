import math


class ID3DecisionTree:
    def __init__(self):
        self.tree = None

    def calculate_entropy(self, labels):
        """
        计算给定标签列表的熵
        """
        num_entries = len(labels)
        label_counts = {}
        for label in labels:
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1
        ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            ent -= prob * math.log(prob, 2)
        return ent

    def split_data(self, data, axis, value):
        """
        根据指定属性和值划分数据集
        """
        ret_data = []
        for feat_vec in data:
            if feat_vec[axis] == value:
                reduced_feat_vec = feat_vec[:axis]
                reduced_feat_vec.extend(feat_vec[axis + 1:])
                ret_data.append(reduced_feat_vec)
        return ret_data

    def choose_best_feature_to_split(self, data):
        """
        选择最佳划分特征
        """
        num_features = len(data[0]) - 1  # 最后一个特征是标签
        base_entropy = self.calculate_entropy([example[-1] for example in data])
        best_info_gain = 0.0
        best_feature = -1
        for i in range(num_features):
            feat_list = [example[i] for example in data]
            unique_vals = set(feat_list)
            new_entropy = 0.0
            for value in unique_vals:
                sub_data = self.split_data(data, i, value)
                prob = len(sub_data) / float(len(data))
                new_entropy += prob * self.calculate_entropy([example[-1] for example in sub_data])
            info_gain = base_entropy - new_entropy
            if (info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    def majority_cnt(self, classlist):
        """
        返回出现次数最多的类别
        """
        classcount = {}
        for vote in classlist:
            if vote not in classcount.keys(): classcount[vote] = 0
            classcount[vote] += 1
        sortedclasscount = sorted(classcount.items(), key=lambda item: item[1], reverse=True)
        return sortedclasscount[0][0]

    def create_tree(self, data, labels):
        """
        构建决策树
        """
        classlist = [example[-1] for example in data]
        if classlist.count(classlist[0]) == len(classlist):
            return classlist[0]  # 所有类别相同，返回该类别
        if len(data[0]) == 1:  # 没有特征，返回类别最多的值
            return self.majority_cnt(classlist)
        best_feat = self.choose_best_feature_to_split(data)
        best_feat_label = labels[best_feat]
        my_tree = {best_feat_label: {}}
        del (labels[best_feat])
        feat_values = [example[best_feat] for example in data]
        unique_values = set(feat_values)
        for value in unique_values:
            sublabels = labels[:]
            my_tree[best_feat_label][value] = self.create_tree(self.split_data(data, best_feat, value), sublabels)
        return my_tree

    def print_tree(self, tree, labels, indent=''):
        """
        打印决策树
        """
        for feat in tree:
            print(f"{indent}{feat}:")
            for value in tree[feat]: