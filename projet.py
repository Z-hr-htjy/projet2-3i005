import pandas as pd
import numpy as np
import scipy.stats as stats
import utils

# 1.1
def getPrior(data):
    n_class1 = sum(data['target'] == 1)
    n_total = len(data)
    p_hat = n_class1 / n_total

    z = stats.norm.ppf(0.975)
    ci_half_width = z * np.sqrt((p_hat * (1 - p_hat)) / n_total)

    return {
        'estimation': p_hat,
        'min5pourcent': p_hat - ci_half_width,
        'max5pourcent': p_hat + ci_half_width
    }

# 1.2
class APrioriClassifier(utils.AbstractClassifier):
    def __init__(self):
        super().__init__()
        self.majority_class = 0

    def estimClass(self, attrs):
        return self.majority_class

    def statsOnDF(self, data):
        self.majority_class = getPrior(data)['estimation'] > 0.5
        VP, VN, FP, FN = 0, 0, 0, 0

        for i in range(len(data)):
            dic = utils.getNthDict(data, i)
            true_class = dic['target']
            predicted_class = self.estimClass(dic)

            if true_class == 1:
                if predicted_class == 1:
                    VP += 1
                else:
                    FN += 1
            else:
                if predicted_class == 1:
                    FP += 1
                else:
                    VN += 1

        precision = VP / (VP + FP) if VP + FP > 0 else 0
        recall = VP / (VP + FN) if VP + FN > 0 else 0

        return {
            'VP': VP, 'VN': VN,
            'FP': FP, 'FN': FN,
            'precision': precision,
            'recall': recall
        }

# 2.1
def P2D_l(df, attr):
    result = {0: {}, 1: {}}

    unique_attr_values = df[attr].unique()
    unique_targets = df['target'].unique()

    for target_value in unique_targets:
        subset = df[df['target'] == target_value]
        total_count = len(subset)

        for attr_value in unique_attr_values:
            count = len(subset[subset[attr] == attr_value])
            result[target_value][attr_value] = count / total_count

    return result


def P2D_p(df, attr):
    result = {}

    unique_attr_values = df[attr].unique()
    unique_targets = df['target'].unique()

    for attr_value in unique_attr_values:
        subset = df[df[attr] == attr_value]
        total_count = len(subset)
        result[attr_value] = {}

        for target_value in unique_targets:
            count = len(subset[subset['target'] == target_value])
            result[attr_value][target_value] = count / total_count

    return result

# 2.2
class ML2DClassifier(APrioriClassifier):
    def __init__(self, df, attr):
        super().__init__()

        self.P2Dl_table = P2D_l(df, attr)
        self.attribute = attr

    def estimClass(self, attrs):
        attr_value = attrs[self.attribute]

        likelihood_0 = self.P2Dl_table[0].get(attr_value, 0)
        likelihood_1 = self.P2Dl_table[1].get(attr_value, 0)

        return 1 if likelihood_1 > likelihood_0 else 0

# 2.3
class MAP2DClassifier(APrioriClassifier):
    def __init__(self, df, attr):
        super().__init__()

        self.P2Dp_table = P2D_p(df, attr)
        self.attribute = attr

    def estimClass(self, attrs):
        attr_value = attrs[self.attribute]

        post_prob_0 = self.P2Dp_table.get(attr_value, {}).get(0, 0)
        post_prob_1 = self.P2Dp_table.get(attr_value, {}).get(1, 0)

        return 1 if post_prob_1 > post_prob_0 else 0

#####
# Question 2.4 : comparaison
#####
# Nous préférons le ML2DClassifier car il offre la meilleure précision.
# Cependant, le choix entre ML2DClassifier et MAP2DClassifier dépend des coûts des erreurs.
# APrioriClassifier, classant tout comme positif, est le moins efficace.
#####

# 3.1
def nbParams(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()

    unique_values = [len(df[col].unique()) for col in columns]

    total_entries = 1
    for value in unique_values:
        total_entries *= value

    octets = total_entries * 8

    memory_size = octets
    gigabytes = memory_size // (1024 ** 3)
    memory_size %= (1024 ** 3)
    megabytes = memory_size // (1024 ** 2)
    memory_size %= (1024 ** 2)
    kilobytes = memory_size // 1024
    memory_size %= 1024

    if gigabytes:
        print(f"{len(columns)} variable(s) : {octets} octets = {gigabytes}go {megabytes}mo {kilobytes}ko {memory_size}o")
    elif megabytes:
        print(f"{len(columns)} variable(s) : {octets} octets = {megabytes}mo {kilobytes}ko {memory_size}o")
    elif kilobytes:
        print(f"{len(columns)} variable(s) : {octets} octets = {kilobytes}ko {memory_size}o")
    else:
        print(f"{len(columns)} variable(s) : {memory_size} octets")

    return octets

# 3.2
def nbParamsIndep(df):
    octets = sum([len(df[col].unique()) for col in df.columns]) * 8

    memory_size = octets
    gigabytes = memory_size // (1024 ** 3)
    memory_size %= (1024 ** 3)
    megabytes = memory_size // (1024 ** 2)
    memory_size %= (1024 ** 2)
    kilobytes = memory_size // 1024
    memory_size %= 1024

    if gigabytes:
        print(f"{df.shape[1]} variable(s) : {octets} octets = {gigabytes}go {megabytes}mo {kilobytes}ko {memory_size}o")
    elif megabytes:
        print(f"{df.shape[1]} variable(s) : {octets} octets = {megabytes}mo {kilobytes}ko {memory_size}o")
    elif kilobytes:
        print(f"{df.shape[1]} variable(s) : {octets} octets = {kilobytes}ko {memory_size}o")
    else:
        print(f"{df.shape[1]} variable(s) : {memory_size} octets")

    return octets

#####
# Question 3.3.a : preuve
#####
# On nous donne que A est indépendant de C sachant B. Cela signifie : P(A,C∣B)=P(A∣B)×P(C∣B)
# On sait aussi, à partir de la définition de la probabilité conditionnelle : P(A,B,C)=P(A,C∣B)×P(B)
# En combinant ces deux équations, on obtient : P(A,B,C)=P(A∣B)×P(C∣B)×P(B)
#####

#####
# Question 3.3.b : complexité en indépendance partielle
#####
# Sans l'utilisation de l'indépendance conditionnelle, la mémoire nécessaire est de 1000 octets.
# Avec l'utilisation de l'indépendance conditionnelle, la mémoire nécessaire est de 440 octets.
#####

#####
# Question 4.1 : Exemples
#####
# Complètement indépendantes
# utils.drawGraphHorizontal("A;B;C;D;E")
# Sans aucune indépendance
# utils.drawGraphHorizontal("A->B->C->D->E")
#####

#####
# Question 4.2 : Naïve Bayes
#####
# Écrire comment se décompose la vraisemblance  P(attr1,attr2,attr3,⋯|target).
# En supposant l'indépendance conditionnelle des attributs étant donné 'target', la vraisemblance se décompose ainsi :
# P(attr1, attr2, attr3, ..., | target) = P(attr1 | target) * P(attr2 | target) * P(attr3 | target) * ...

# Écrire comment se décompose la distribution a posteriori  P(target|attr1,attr2,attr3,⋯).
# D'après le théorème de Bayes, P(target | attr1, attr2, attr3, ...) est proportionnelle à P(attr1, attr2, attr3, ..., | target) * P(target).
# En utilisant l'indépendance conditionnelle des attributs étant donné 'target' dans le modèle Naïve Bayes, cela devient :
# P(target | attr1, attr2, attr3, ...) ∝ P(attr1 | target) * P(attr2 | target) * P(attr3 | target) * ... * P(target).
# Ce qui signifie que cette distribution a posteriori est proportionnelle au produit des probabilités conditionnelles des attributs étant donné la classe cible, multiplié par la probabilité a priori de cette classe.
#####

# 4.3
def drawNaiveBayes(df, class_column):
    attributes = [col for col in df.columns if col != class_column]
    graph_links = ";".join([f"{class_column}->{attr}" for attr in attributes])
    return utils.drawGraph(graph_links)


def nbParamsNaiveBayes(df, class_column, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    if len(columns) == 0:
        columns = [class_column]
        l_columns = 0
    else:
        l_columns = len(columns)

    unique_class_values = df[class_column].nunique()

    octets = unique_class_values * 8

    for col in columns:
        if col != class_column:
            octets += df[col].nunique() * unique_class_values * 8

    memory_size = octets
    gigabytes = memory_size // (1024 ** 3)
    memory_size %= (1024 ** 3)
    megabytes = memory_size // (1024 ** 2)
    memory_size %= (1024 ** 2)
    kilobytes = memory_size // 1024
    memory_size %= 1024

    if gigabytes:
        print(f"{l_columns} variable(s) : {octets} octets = {gigabytes}go {megabytes}mo {kilobytes}ko {memory_size}o")
    elif megabytes:
        print(f"{l_columns} variable(s) : {octets} octets = {megabytes}mo {kilobytes}ko {memory_size}o")
    elif kilobytes:
        print(f"{l_columns} variable(s) : {octets} octets = {kilobytes}ko {memory_size}o")
    else:
        print(f"{l_columns} variable(s) : {memory_size} octets")

    return octets

# 4.4
class NaiveBayesClassifier(APrioriClassifier):
    def __init__(self, df, class_col='target'):
        super().__init__()
        self.df = df
        self.class_col = class_col
        self.classes = sorted(df[class_col].unique())
        self.params = {}
        for cls in self.classes:
            self.params[cls] = {}
            class_df = df[df[class_col] == cls]
            for col in df.columns:
                if col != class_col:
                    self.params[cls][col] = class_df[col].value_counts(normalize=True).to_dict()

    def estimProbas(self, x):
        pass

    def estimClass(self, x):
        probas = self.estimProbas(x)
        return max(probas, key=probas.get)


class MLNaiveBayesClassifier(NaiveBayesClassifier):
    def estimProbas(self, x):
        probas = {}
        for cls in self.classes:
            probas[cls] = 1
            for attr, value in x.items():
                if attr != self.class_col and attr in self.params[cls].keys():
                    probas[cls] *= self.params[cls][attr].get(value, 0)
        return probas


class MAPNaiveBayesClassifier(NaiveBayesClassifier):
    def __init__(self, df, class_col='target'):
        super().__init__(df, class_col)
        prior = getPrior(df)['estimation']
        self.class_prob = {1:prior, 0:1-prior}

    def estimProbas(self, x):
        probas = {}
        for cls in self.classes:
            probas[cls] = self.class_prob[cls]
            for attr, value in x.items():
                if attr != self.class_col and attr in self.params[cls].keys():
                    probas[cls] *= self.params[cls][attr].get(value, 0)
        sum_probas = sum(probas.values())
        if sum_probas != 0:
            for cls in probas:
                probas[cls] /= sum_probas
        return probas

# 5.1
from scipy.stats import chi2_contingency

def isIndepFromTarget(df, attr, x):
    contingency = pd.crosstab(df[attr], df['target'])
    _, p, _, _ = chi2_contingency(contingency)
    return p > x

# 5.2
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    def __init__(self, df, x):
        self.independent_attrs = [attr for attr in df.keys() if attr != 'target' and isIndepFromTarget(df, attr, x)]
        super().__init__(df.drop(columns=self.independent_attrs))

    def draw(self):
        graph_str = 'target->' + ';target->'.join([col for col in self.df.columns if col != 'target'])
        return utils.drawGraph(graph_str)

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df, x):
        self.independent_attrs = [attr for attr in df.keys() if attr != 'target' and isIndepFromTarget(df, attr, x)]
        super().__init__(df.drop(columns=self.independent_attrs))

    def draw(self):
        graph_str = 'target->' + ';target->'.join([col for col in self.df.columns if col != 'target'])
        return utils.drawGraph(graph_str)

#####
# Question 6.1 : Evaluation des classifieurs
#####
# Le point idéal se trouve à la coordonnée (1,1).
# Visualiser la distance de chaque point classifieur au point idéal (1,1). Le classifieur le plus proche de ce point
# serait le meilleur en termes de compromis précision-rappel.
#####

import matplotlib.pyplot as plt

def mapClassifiers(dic, df):
    precisions = []
    recalls = []
    names = []

    for name, classifier in dic.items():
        stats = classifier.statsOnDF(df)
        precisions.append(stats["precision"])
        recalls.append(stats["recall"])
        names.append(name)

    for i, name in enumerate(names):
        plt.scatter(precisions[i], recalls[i], marker='x', color='red')
        plt.annotate(name, (precisions[i], recalls[i]))
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(min(precisions) - 0.01, max(precisions) + 0.01)
    plt.ylim(min(recalls) - 0.01, max(recalls) + 0.01)
    plt.show()

#####
# Question 6.3 : Conclusion
#####
# 1. Les bons résultats sur le jeu d'entraînement ne garantissent pas les mêmes sur le jeu de test.
# 2. Il y a un compromis observable entre la précision et le rappel.
# 3. ML2DClassifier semble être le plus performant en considérant ces deux mesures.
#####

# 7.1
def MutualInformation(df, x, y):
    values_x = df[x].unique()
    values_y = df[y].unique()

    prob_x = {value: len(df[df[x] == value]) / len(df) for value in values_x}
    prob_y = {value: len(df[df[y] == value]) / len(df) for value in values_y}

    mi = 0
    for val_x in values_x:
        for val_y in values_y:
            joint_prob = len(df[(df[x] == val_x) & (df[y] == val_y)]) / len(df)
            if joint_prob > 0:
                mi += joint_prob * np.log2(joint_prob / (prob_x[val_x] * prob_y[val_y]))
    return mi

def ConditionalMutualInformation(df, x, y, z):
    values_x = df[x].unique()
    values_y = df[y].unique()
    values_z = df[z].unique()

    cmi = 0
    for val_z in values_z:
        prob_z = len(df[df[z] == val_z]) / len(df)
        prob_x_z = {value: len(df[(df[x] == value) & (df[z] == val_z)]) / len(df) for value in values_x}
        prob_y_z = {value: len(df[(df[y] == value) & (df[z] == val_z)]) / len(df) for value in values_y}

        for val_x in values_x:
            for val_y in values_y:
                joint_prob = len(df[(df[x] == val_x) & (df[y] == val_y) & (df[z] == val_z)]) / len(df)
                if joint_prob > 0:
                    cmi += joint_prob * np.log2(joint_prob * prob_z / (prob_x_z[val_x] * prob_y_z[val_y]))
    return cmi

# 7.2
def MeanForSymetricWeights(a):
    n = a.shape[0]
    total = sum([a[i, j] for i in range(n) for j in range(i+1, n)])
    return total / (n * (n-1) / 2)

def SimplifyConditionalMutualInformationMatrix(a):
    threshold = MeanForSymetricWeights(a)
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            if a[i, j] < threshold:
                a[i, j] = 0
    return a

# 7.3
def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]


def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[xroot] = yroot
        rank[yroot] += 1


def Kruskal(df, a):
    keys = list(df.keys())
    n = len(keys) - 1
    parent = [i for i in range(n)]
    rank = [0] * n

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, a[i, j]))

    edges.sort(key=lambda x: x[2], reverse=True)
    mst = []
    for i, j, w in edges:
        if find(parent, i) != find(parent, j) and w > 0.25:
            mst.append((keys[i], keys[j], w))
            union(parent, rank, i, j)

    return mst

# 7.4
def ConnexSets(list_arcs):
    sets = []
    def find_set(a):
        for s in sets:
            if a in s:
                return s
        return None

    for arc in list_arcs:
        a, b, _ = arc
        set_a = find_set(a)
        set_b = find_set(b)

        if set_a is None and set_b is None:
            sets.append({a, b})
        elif set_a is None:
            set_b.add(a)
        elif set_b is None:
            set_a.add(b)
        elif set_a != set_b:
            set_a.update(set_b)
            sets.remove(set_b)

    return sets


def OrientConnexSets(df, arcs, classe):
    connex_sets = ConnexSets(arcs)
    oriented_arcs = []

    for connex_set in connex_sets:
        roots = [max(connex_set, key=lambda a: MutualInformation(df, a, classe))]
        for _ in arcs:
            for arc in arcs:
                a, b, _ = arc
                if a in connex_set and b in connex_set:
                    if a in roots and b not in roots:
                        oriented_arcs.append((a, b))
                        roots.append(b)
                    if b in roots and a not in roots:
                        oriented_arcs.append((b, a))
                        roots.append(a)

    return oriented_arcs

# 7.5
class MAPTANClassifier(APrioriClassifier):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.n = len(df)
        self.class_name = "target"
        self.classes = sorted(df[self.class_name].unique())

        cmis = np.array([[0 if x == y else ConditionalMutualInformation(df, x, y, self.class_name)
                          for x in df.keys() if x != self.class_name]
                         for y in df.keys() if y != self.class_name])
        SimplifyConditionalMutualInformationMatrix(cmis)
        arcs = Kruskal(df, cmis)
        self.arcs = OrientConnexSets(df, arcs, self.class_name)
        prior = getPrior(df)['estimation']
        self.class_prob = {1:prior, 0:1-prior}

    def draw(self):
        arcs_string = ";".join([f"{self.class_name}->{a}" for a in self.df.keys() if a != self.class_name] + [f"{a}->{b}" for a, b in self.arcs])
        return utils.drawGraph(arcs_string)

    def estimProbas(self, data):
        probas = {}
        for cls in self.classes:
            prob = self.class_prob[cls]

            for attr, val in data.items():
                if attr == self.class_name:
                    continue

                parent = next((a for a, b in self.arcs if b == attr), None)
                if parent:
                    parent_val = data[parent]
                    num = len(self.df[(self.df[attr] == val) & (self.df[parent] == parent_val) & (self.df[self.class_name] == cls)])
                    den = len(self.df[(self.df[parent] == parent_val) & (self.df[self.class_name] == cls)])
                else:
                    num = len(self.df[(self.df[attr] == val) & (self.df[self.class_name] == cls)])
                    den = len(self.df[self.df[self.class_name] == cls])
                prob *= (num + 1) / (den + 12)

            probas[cls] = prob

        total = sum(probas.values())
        if total != 0:
            for cls in self.classes:
                probas[cls] /= total

        return probas

    def estimClass(self, d):
        probas = self.estimProbas(d)
        return max(probas.keys(), key=(lambda k: probas[k]))

#####
# Question 8 : Conclusion finale
#####
# Les classifieurs bayésiens sont robustes et interprétables.
# La sophistication, comme avec TAN, peut améliorer la performance,
# mais attention à la suradaptation.
# Il est crucial de comparer différents modèles pour chaque tâche.
#####