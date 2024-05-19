# 基于GNN的系统架构模型搜索技术报告

## 需求及问题

### 需求概述

本项目旨在开发一个系统，通过搜索数据库中的IP库和IP设计库，对不完整的基于IP的架构设计进行搜索推荐。具体目标是利用数据库中现有的数据，寻找相似度最高的设计，并对不完整的架构进行补全。

### 需要解决的问题

1. 对输入架构定义模型进行抽象，抽象成具有节点属性和边属性的有向图。
2. 研究图搜索算法，给出最准确的相似图推荐。
3. 验证搜索准确率。
4. 应用于系统架构模型搜索。

### 存在问题，及解决方案

#### 存在问题

+ 目前只有少量IP库和极少量的IP设计库，难以直接面向项目解决问题。
+ 对搜索效果的评估困难，难以鉴定给出的推荐效果，很难对推荐模型给出评估。

#### 拟解决方案

+ 先将架构抽象成有向图，用现有的公开模型数据集，将公开数据集抽象成有向图形式，并用json存储模型图结构，在这个有向图结构数据集的基础上，展开图搜索匹配算法的设计和实验。
+ 目前拟采用随机mask的测试构造方案，在给出的top_n推荐中包含ground truth则认为推荐准度为1，否则为0。

## 国内外相关技术研究进展

### 模型驱动工程领域的推荐系统综述

#### 算法分类

 可以根据它们用于计算相关性的算法方法进行分类。在这方面，我们可以区分出两种类型的系统：

1. **基于存储的推荐：**通过启发式方法评估模型的相似度完成推荐。
2. **基于模型的推荐系统：**它们通过使用通过机器学习技术（例如矩阵分解或神经网络）构建的基于数据的模型来预测物品相关性。

#### 推荐系统的分类及其优缺点

##### 内容推荐系统（Content-Based Recommenders）

**解释**：内容推荐系统根据用户过去喜欢的项目的特征来推荐相似的项目。它使用项目的属性或特征（如关键词、元数据和标签）来表示用户和项目的档案，并基于这些特征来计算用户和项目之间的相似性。

**优点**：

- **个性化推荐**：能够提供基于用户历史偏好的个性化推荐。
- **冷项目处理**：可以推荐尚未被任何用户评价过的新项目（冷项目），因为推荐基于项目内容的相似性。

**缺点**：

- **过度专业化**：容易陷入过度推荐相似项目的问题，限制了用户发现新颖或意外（偶然性）的项目。
- **新用户冷启动问题**：需要大量用户偏好信息才能提供准确的推荐，新用户在初始阶段的推荐质量较低。

##### 协同过滤推荐系统（Collaborative Filtering Recommenders）

**解释**：协同过滤推荐系统根据相似用户的偏好来推荐项目。它依赖于用户对项目的反馈（通常是评分），通过用户和项目的评分相似性和模式来建立推荐模型。

**优点**：

- **多样化推荐**：能够提供新颖和多样化的推荐，用户可以发现之前未接触过的项目。
- **实际应用广泛**：在许多实际应用中表现优异，是最广泛使用的个性化推荐方法。

**缺点**：

- **新用户冷启动问题**：需要足够的用户评分信息才能提供准确的推荐，新用户在初始阶段的推荐质量较低。
- **新项目冷启动问题**：一个项目只有在被评分后才能被推荐，新项目在初始阶段的推荐频率较低。
- **高稀疏性问题**：当评分数据稀疏时，推荐质量下降。

##### 知识推荐系统（Knowledge-Based Recommenders）

**解释**：知识推荐系统利用领域特定的知识来描述和关联用户和项目，以提供个性化推荐。常见的实现方法包括基于案例推理和基于约束的方法。

**优点**：

- **准确推荐**：能够提供和解释准确的推荐，特别是当需要深入理解用户需求时。
- **无冷启动问题**：不依赖于用户的历史偏好，因此不会受到新用户或新项目冷启动问题的影响。

**缺点**：

- **知识获取瓶颈**：需要专家知识来构建和维护知识库，成本高且过程复杂。
- **适用性限制**：通常是特定问题的专用解决方案，难以推广到其他领域或问题。

##### 混合推荐系统（Hybrid Recommenders）

**解释**：混合推荐系统结合两种或多种推荐方法，以弥补单一方法的不足。例如，结合内容推荐和协同过滤推荐，以提高推荐质量和覆盖面。

**优点**：

- **综合优势**：能够结合不同推荐方法的优点，提供更全面和准确的推荐。
- **降低冷启动问题**：通过结合多种方法，可以减轻新用户和新项目的冷启动问题。

**缺点**：

- **实现复杂性**：实现混合推荐系统通常比单一推荐系统复杂，可能需要更多的计算资源和开发时间。

#### 推荐系统的评估

##### 1. 离线实验（Offline Experiments）

**解释**：

- 离线实验使用历史数据集进行模拟评估。通过将数据集划分为训练集和测试集，训练集用于构建推荐模型，测试集用于评估模型的性能。

**常用评估指标**：

- **精确度（Precision）**：推荐结果中相关项目占推荐项目总数的比例。
- **召回率（Recall）**：推荐结果中相关项目占所有相关项目总数的比例。
- **F值（F-measure）**：精确度和召回率的调和平均数，用于综合评价推荐系统的性能。
- **均方根误差（RMSE）**：预测评分与实际评分之间的差值的平方的平均数的平方根，用于评价评分预测的准确性。

**优点**：

- **可重复性**：实验可在不同的环境中重复进行，结果具有较高的可信度。
- **成本低**：不需要真实用户参与，实验成本相对较低。

**缺点**：

- **用户体验欠缺**：无法反映真实用户的体验和反馈，仅依赖历史数据进行评估。

##### 2. 在线实验（Online Experiments）

**解释**：

- 在线实验在实际场景中进行用户评估。通过在真实环境中部署推荐系统，并让用户使用系统，收集用户的行为数据和反馈信息。

**常用方法**：

- **A/B测试**：将用户随机分成两组，一组使用现有系统（A版本），另一组使用新系统（B版本），比较两组用户的行为和反馈，评估新系统的效果。
- **多臂老虎机实验**：一种动态调整实验组的方法，根据实时反馈调整不同版本的曝光比例，优化推荐效果。

**优点**：

- **真实反馈**：能够获得真实用户的反馈，评估结果更具参考价值。
- **用户参与**：通过用户的实际使用行为，评估推荐系统在真实环境中的表现。

**缺点**：

- **成本高**：需要在实际环境中部署系统，并进行长时间的用户数据收集，成本较高。
- **受限条件**：实验可能受到用户数量、时间和环境等因素的限制，影响结果的普遍性。

##### 3. 用户研究（User Studies）

**解释**：

- 用户研究在受控环境中进行，小规模用户组实验，收集用户反馈和使用体验。通常通过问卷调查、访谈和实验观察等方法获取用户对推荐系统的主观评价。

**常用方法**：

- **问卷调查**：通过设计问卷，获取用户对推荐系统的满意度、易用性和接受度等信息。
- **用户访谈**：与用户进行一对一访谈，深入了解用户的需求、使用体验和改进建议。
- **实验观察**：在实验室环境中观察用户使用推荐系统的行为，分析用户的操作习惯和问题。

**优点**：

- **深入分析**：能够深入了解用户的主观体验和意见，获取详细的反馈信息。
- **灵活性高**：实验设计灵活，可以根据需要调整实验内容和方法。

**缺点**：

- **样本有限**：通常样本量较小，结果可能不具有普遍性。
- **成本较高**：需要投入时间和人力进行实验设计、数据收集和分析，成本较高。

#### 推荐系统在MDE领域的用途

##### 1. 模型完成（Model Completion）

**解释**：辅助用户完成部分模型，通过提供建议补全模型元素，使得部分模型能够达到完整和正确的状态。

**用途**：

- 帮助用户在建模过程中快速找到和补充所需的模型元素。
- 提供智能建议，减少用户的工作量和错误率。

**相关工作**：SimVMA 、Heinemann 、Kögel et al. 。

##### 2. 模型修复（Model Repair）

**解释**：检测和修复模型中的错误或不一致之处，通过提供修复建议，使得模型符合预定义的约束和规则。

**用途**：

- 帮助用户发现和修复模型中的错误，保证模型的正确性和一致性。
- 提供自动化修复工具，提高修复效率和准确性。

**相关工作**：IntellEdit 、PARMOREL 、DPF 

##### 3. 模型创建（Model Creation）

**解释**：从头开始创建模型，提供指导和建议，帮助用户构建符合需求的模型。

**用途**：

- 帮助新手用户快速上手建模，提供建模流程和步骤的建议。
- 提供智能模板和示例，加速模型创建过程。

##### 4. 模型重用（Model Reuse）

**解释**：帮助用户找到和重用已有的模型片段，减少重复工作，提高建模效率。

**用途**：

- 提供模型库搜索和推荐工具，帮助用户快速找到相关的模型片段。
- 提供重用建议，指导用户将已有模型片段整合到新模型中。

**相关工作**：DoMoRe

##### 5. 模型查找（Model Find）

**解释**：帮助用户在大规模模型库中查找相关模型或模型片段，提高模型检索效率。

**用途**：

- 提供智能搜索工具，帮助用户快速定位所需的模型或片段。
- 提供推荐功能，根据用户需求和偏好推荐相关模型。

#### 模型驱动领域推荐系统相关研究统计

1. 推荐目的
   - 主要集中在模型的完成和修复任务上，分别占比 73.4% 和 10.9%。
2. 推荐方法
   - 知识推荐系统（47%）、内容推荐系统（19.6%）、混合推荐系统（11.8%）、协同过滤推荐系统（7.8%）。
3. 评估方法
   - 离线实验最为常见，占比 37%，其次是用户研究，在线实验较少。

### 相关研究：Morgan

#### 数据集

Mogan对元模型补全，通过添加新的classes, attributes和relations来完成补全，具体抽象提取元模型方法为：编写了一个ecore 解析器，从元模型开始，Ecore解析器提取元类及其结构特征列表，即属性和引用。特别是，该组件通过从根包及其子包开始浏览树结构以查找所提到的元素。关于结构特征，每个元素表示为以下定义的三元组。

• Name: 元素的名称。

• Type: 元素使用某种类型进行特征化。例如，从 ecore 元模型提取的规范类型可以是 ESTRING 或 EINT。

•  Relation: 最后一个组件标识元素与类之间的关系类型，即元模型为属性/引用，模型为方法/字段。

##### 元模型：

![image-20240519164948370](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519164948370.png)

**元模型三元组表示：**Ecore解析器提取元类及其结构特征的列表，即属性和引用。具体来说，元类名称标识相应的元类实例，而每个关系表示为以下定义的三元组：

```
metaclass-name (〈Name, Type, [reference | attribute]〉)∗，例如：Actor (name, EString, attribute)

// Name是结构特征的名称，Type标识类型元素，例如，EString或EInt，Relation识别元素与类之间的关系类型，即属性或引用。
```

***元模型推荐，Morgan数据集实例：***

```
uml	TypedTransition (parameters,Parameter,ref) (guards,Guard,ref) 
uml	Automaton (eventPatternId,EString,attribute) (initialState,InitState,ref) (finalStates,FinalState,ref) (eventTokens,EventToken,ref) (timedZones,TimedZone,ref) (states,State,ref) (trapState,TrapState,ref) 
uml	ParameterTable (eventToken,EventToken,ref) (parameterBindings,ParameterBinding,ref) 
uml	State (label,EString,attribute) (outStateOf,TimedZone,ref) (inStateOf,TimedZone,ref) (lastProcessedEvent,null,ref) (inTransitions,Transition,ref) (outTransitions,Transition,ref) (eventTokens,EventToken,ref) 
uml	Transition (preState,State,ref) (postState,State,ref) 
uml	Parameter (symbolicName,EString,attribute) (position,EInt,attribute) (transition,TypedTransition,ref) 
uml	Guard (eventType,null,ref) (transition,TypedTransition,ref) 
uml	InternalModel (latestEvent,null,ref) (automata,Automaton,ref) (enabledForTheLatestEvent,Automaton,ref) (eventTokensInModel,EventToken,ref) 
uml	ParameterBinding (symbolicName,EString,attribute) (value,EJavaObject,attribute) (parameterTable,ParameterTable,ref) 
uml	EventToken (lastProcessed,null,ref) (timedZones,TimedZone,ref) (currentState,State,ref) (parameterTable,ParameterTable,ref) (recordedEvents,null,ref) 
uml	TimedZone (time,ELong,attribute) (outState,State,ref) (inState,State,ref)
```

##### 模型：

![image-20240519165257504](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519165257504.png)

每个MoDisco模型被表示为一个Java类列表，后跟方法定义和字段声明。类名表示实例标识符，而每个关系表示为以下定义的三元组。

**模型三元组表示：**

```
class-name (〈Name, Return type, [method | field]〉)∗，例如：LateBinding (getError, EString, method).

// 其中，名称（Name）是方法或字段的名称，返回类型（Return type）是方法或字段的类型，关系（Relation）标识类成员之间的关系类型，即方法或字段与类之间的关系。
```

***Java类图模型推荐，Morgan数据集实例：***

```
spring	RouteTemplateContextDefinitionParser (doParse,void,method) 
spring	RestContextDefinitionParser (doParse,void,method) 
spring	BridgePropertiesParser (parseUri,String,method) (parseProperty,String,method) (delegate,org.apache.camel.component.properties.PropertiesParser,field) (parser,org.apache.camel.component.properties.PropertiesParser,field) 
spring	RouteContextDefinitionParser (doParse,void,method) 
spring	SSLContextParametersFactoryBeanBeanDefinitionParser (doParse,void,method) 
spring	EndpointDefinitionParser (doParse,void,method) 
spring	CamelContextBeanDefinitionParser (doParse,void,method) 
spring	RedeliveryPolicyDefinitionParser (shouldGenerateId,boolean,method) 
spring	BridgePropertyPlaceholderResolver (resolvePlaceholder,String,method) (properties,org.apache.camel.component.properties.PropertiesLookup,field)
```



## 研究思路和方案

### I/O

+ 输入：不完整的活动图、数据流图形式的模型（形式待定）。
+ 输出：从现有模型库中搜索到的，相似的较完整的活动图、数据流图模型。

### 数据集和构建方法

#### ModelSet公开数据集

##### 主要特点：

1. 它包含5,000多个Ecore模型(从GitHub中提取)和5,000多个UML模型(从GenMyModel中提取)。
2. 这些模型已经用其类别进行了标记，该类别代表共享相似应用程序域的一种模型类型。以下图表包含主要类别的摘要(从中可以看出人们喜欢构建 state machine 元模型和描述 shopping 域的UML模型！)

​	![image-20240519144611705](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519144611705.png)                       

3. 此外，ModelSet还包含其他提供更多语义信息的标签。例如，可以用 category: statemachine 、 timed 值的附加标签和另一个 teaching 来表示该特定模型正用于教学目的。总共有28,000多个品牌

##### modelset数据集潜在应用场景：

1. 以标签为目标变量的分类方法。
2. 推荐系统(例如，建议属性名称/类型等)
3. 对聚类方法的评估，其中标签提供关于聚类的ground truth。
4. 伪模型识别(即，使用“伪”标签)
5. 通过使用标签以分层方式进行训练-测试-评估拆分来评估ML模型。
6. 基于标签的训练嵌入(即，用于使用结果向量的克隆检测)。
7. 模型域的实证分析。

##### 应用（ [MAR - The model search platform (mar-search.org)](http://mar-search.org/)）：

![image-20240519144840694](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519144840694.png)

##### 数据集结构

数据集基本上包括：

1. 原始模型
2. 具有关于模型的标签和信息的数据库，
3. 模型的图结构表示，
4. 模型的替代序列化(例如，作为文本文件)。

以下是解压包时会发现的结构。

```
[+] datasets
    [+] dataset.ecore
        [+] data 
	        [+] ecore.db
		        - The database with the labels for the Ecore models
	        [+] analysis.db 
		        - Statistics about the Ecore models
    [+] dataset.genmymodel
	        [+] genmymodel.db
		        - The database with the labels for the UML models
	        [+] analysis.db
		        - Statistics about the UML models
[+] raw-data
    [+] repo-ecore-all
        - The .ecore models that has been labelled
    [+] repo-genmymodel-uml
        - The UML models that has been labelled, stored as .xmi files 
[+] txt
    - A mirror of raw-data but with 1-gram encoding of the models,
	  that is for each model a textual file with the strings of the model.
[+] graph

```

其中，模型的图表示以json格式存储在graph文件夹中，模型原文件以xmi形式存储在raw-data文件夹中。

##### json表示的有向图模型

![顶点集](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519144217144.png)![边集](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519144312223.png)

##### 数据集优缺点总结

+ 缺点：没有针对性的活动图数据集（只有30个左右活动图模型），并且图结构中属性信息极少有待从其模型源文件中挖掘信息。
+ 优点：提供了一套完整的数据集demo，包括元模型的表示与存储，元模型的图结构json表示，以及用SQLite数据库存储模型的label信息。

#### 测试数据集构建思路

*这里我们默认将输入抽象成有向图，用json存储这种有向图结构。*

收集ModelSet数据集中的所有模型的图表示文件（.json）约5000个文件，这些文件既是训练集也是ground truth，从中随机选取一部分随机遮罩形成测试集，用测试集去搜索匹配训练集中的模型，模拟对不完整模型的搜索补全。

这里的遮罩采用不同mask程度进行验证，以验证模型在不同完成度下的匹配效果。

下面是一个不同程度mask的测试集得到的推荐效果：

![image-20240519153134779](C:\Users\98157\AppData\Roaming\Typora\typora-user-images\image-20240519153134779.png)

### 图相似度匹配算法

#### 图核相似度匹配

**Weisfeiler-Lehman图核相似度匹配是一种用于图形数据的算法，旨在通过捕捉图形结构的相似性来进行图形匹配和分类。以下是该方法的基本原理：**

##### 1. 基本概念

+ **图 (Graph)**：由节点（顶点）和连接节点的边组成的结构。
+ **核 (Kernel)**：在机器学习中，核是一个函数，它将数据映射到一个高维空间，以便在该空间中应用线性分类器或回归器。对于图核，它衡量的是图形之间的相似性。

##### 2.  **Weisfeiler-Lehman (WL) 颜色优化过程**

Weisfeiler-Lehman算法最初是用于图同构（即判断两个图是否结构相同）的问题。该算法的基本思想是通过迭代地“染色”节点来生成图的特征表示。

 过程如下：

1. **初始化染色**： 每个节点根据其标签或特征（如果有）被赋予一个初始颜色（标签）。
2. **邻域聚合**： 对于每个节点，收集该节点的邻居节点的颜色，并将这些颜色按某种固定顺序排列形成一个字符串。
3. **重新染色**： 使用一个散列函数（哈希函数）将每个节点的当前颜色和邻居颜色字符串映射到一个新的颜色。这一步骤确保具有相似邻域结构的节点会被映射到相同的或相似的颜色。
4. **迭代**： 重复步骤2和步骤3若干次，直到颜色不再变化或达到预定的迭代次数。

##### 3. **生成图的特征表示**

经过若干次迭代后，每个节点的颜色反映了它在图中的结构位置。通过统计每种颜色的频率，可以得到一个图的特征表示，即所谓的“图谱”。

##### 4. **计算图核**

- **Histogram Intersection Kernel**： 一种简单的图核是比较两个图的颜色直方图。对于每种颜色，计算两个图中该颜色的频率之和，形成一个相似度分数。
- **Random Walk Kernel**： 另一种更复杂的方法是考虑所有可能的节点对之间的随机游走，通过计算它们的相似性来构建图核。

##### 5. **相似度匹配**

通过上述步骤，两个图之间的相似度可以通过它们的特征表示（即颜色分布）进行计算。相似度越高，两个图的结构越相似。

##### 应用

Weisfeiler-Lehman图核相似度匹配广泛应用于化学信息学（如化合物结构比对）、生物信息学（如蛋白质网络分析）、社交网络分析等领域。

总结来说，Weisfeiler-Lehman图核相似度匹配通过迭代染色和邻域信息聚合的方法，将图的结构特征转化为可比对的特征向量，从而实现图的相似度计算和匹配。

##### **Grakel库**

**grakel（Graph Kernels Library）**是一个用于图核函数的 Python 库，旨在支持图数据的机器学习和数据挖掘任务。该库提供了多种图核函数，这些核函数可用于比较和分析图结构，从而在图分类、图聚类等任务中发挥作用。

**grakel 库的主要特点和用途：**

图核函数： 提供了多种图核函数，如随机游走核、子图核、Weisfeiler-Lehman核等。这些核函数能够捕捉图的结构信息，可用于度量图之间的相似性。

图表示： 提供了一种灵活的图表示形式，称为 Graph 对象，用于在图上执行核函数计算。这个对象可以灵活地处理图的节点、边和标签信息。

图分类： grakel 库可用于图分类任务，其中图被分为不同的类别。通过使用图核函数，可以比较图之间的相似性，从而支持图分类算法的训练和评估。

图聚类： 通过图核函数，grakel 库也可用于图聚类，即将图数据划分为不同的群组，以便在图的结构上发现模式。

#### GNN（研究验证中）

## 预期达到的目标和成果

+ 实验：在给出top_n推荐n=4的情况下达到75%以上的准确率。

+ 专利：总结工作，争取发表一到两篇专利。
+ 论文：研究达到论文发表条件后，总结成小论文发表。

