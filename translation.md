# 2026 MCM Problem C: Data With The Stars
> 2026年美国大学生数学建模竞赛C题：数据与星辰

![](images/ee32243ea83e90a895ef03ed8062774176a4fe6ff8a421db4d9f4f0d61420269.jpg)

Dancing with the Stars (DWTS) is the American version of an international television franchise based on the British show “Strictly Come Dancing” (“Come Dancing” originally). Versions of the show have appeared in Albania, Argentina, Australia, China, France, India, and many other countries. The U.S. version, the focus of this problem, has completed 34 seasons.
> 与星共舞（DWTS）是美国版本的国际电视节目系列，基于英国节目“舞动奇迹”（最初为“舞动起来”）。该节目的版本已出现在阿尔巴尼亚、阿根廷、澳大利亚、中国、法国、印度等许多国家。本问题重点关注的美国版本已完成34季。

Celebrities are partnered with professional dancers and then perform dances each week. A panel of expert judges scores each couple’s dance, and fans vote (by phone or online) for their favorite couple that week. Fans can vote once or multiple times up to a limit announced each week. Further, fans vote for the star they wish to keep, but cannot vote to eliminate a star. The judge and fan votes are combined in order to determine which couple to eliminate (the lowest combined score) that week. Three (in some seasons more) couples reach the finals and in the week of the finals the combined scores from fans and judges are used to rank them from $1 ^ { \mathrm { s t } }$ to $3 ^ { \mathrm { r d } }$ $( \mathrm { o r ~ } 4 ^ { \mathrm { t h } } , 5 ^ { \mathrm { t h } } )$ .
> 明星与专业舞者搭档，每周表演舞蹈。专家评委团为每对搭档的舞蹈评分，粉丝通过电话或在线投票选出当周最喜爱的组合。粉丝每周可投票一次或多次，但次数上限由每周公布的规定决定。此外，粉丝投票支持他们希望留住的明星，但不能投票淘汰某位明星。评委评分和粉丝投票结果相结合，以决定当周淘汰哪对组合（综合得分最低者）。三对（某些赛季更多）组合进入决赛，在决赛周，评委和粉丝的综合得分用于将他们从第一名排至第三名（或第四名、第五名）。

There are many possible methods of combining fan votes and judge scores. In the first two seasons of the U.S. show, the combination was based on ranks. Season 2 concerns (due to celebrity contestant Jerry Rice who was a finalist despite very low judge scores) led to a modification to use percentages instead of ranks. Examples of these two approaches are provided in the Appendix.
> 将粉丝投票与评委评分结合的方法多种多样。在美国版节目的前两季中，排名是结合的依据。第二季中出现的争议（例如名人选手杰里·赖斯尽管评委评分很低却进入了决赛）促使节目改用百分比而非排名。附录中提供了这两种方法的示例。

In season 27, another “controversy” occurred when celebrity contestant Bobby Bones won despite consistently low judges scores. In response, starting in season 28 a slight modification to the elimination process was made. The bottom two contestants were identified using the combined judge scores and fan votes, and then during the live show the judges voted to select which of these two to eliminate. Around this same season, the producers also returned to using the method of ranks to combine judges scores with fan votes as in seasons one and two. The exact season this change occurred is not known, but it is reasonable to assume it was season 28.
> 在第27季，名人选手鲍比·博恩斯尽管评委评分持续偏低却最终获胜，引发了另一场“争议”。作为回应，从第28季开始，淘汰流程进行了微调。结合评委评分和粉丝投票，确定出排名垫底的两位选手，然后在直播节目中由评委投票决定淘汰其中哪一位。大约在同一季，制作方也重新采用了第一季和第二季中使用排名的方法，将评委评分与粉丝投票相结合。这一变化具体发生在哪一季尚不明确，但可以合理推测是在第28季。

Judge scores are meant to reflect which dancers are technically better, although there is some subjectivity in what makes a dance better. Fan votes are likely much more subjective, influenced by the quality of the dance, but also the popularity and charisma of the celebrity. Show producers might actually prefer, to some extent, conflicts in opinions and votes as such occurrences boost fan interest and excitement.
> 评委打分旨在反映哪些舞者在技术上更胜一筹，尽管舞蹈优劣的判断存在一定主观性。而观众投票则可能更具主观色彩，不仅受舞蹈质量影响，也取决于明星的人气与个人魅力。某种程度上，节目制作方或许更乐于见到意见与投票的分歧，因为这类争议能有效提升粉丝的关注度与观赛热情。

Data with judges scores and contestant information is provided and described below. You may choose to include additional information or other data at your discretion, but you must completely document the sources. Use the data to:
> 以下提供了包含评委评分和参赛者信息的数据集及其描述。您可选择酌情添加额外信息或其他数据，但必须完整记录所有数据来源。请基于这些数据完成以下任务：

Develop a mathematical model (or models) to produce estimated fan votes (which are unknown and a closely guarded secret) for each contestant for the weeks they competed. o Does your model correctly estimate fan votes that lead to results consistent with who was eliminated each week? Provide measures of the consistency. o How much certainty is there in the fan vote totals you produced, and is that certainty always the same for each contestant/week? Provide measures of your certainty for the estimates.
> 开发一个（或多个）数学模型，用于估算每位参赛者在参赛各周的粉丝投票数（该数据未知且严格保密）。o 你的模型是否能正确估算出与每周淘汰结果一致的粉丝投票数？请提供一致性度量指标。o 你生成的粉丝投票总数有多少确定性？这种确定性对每位参赛者/每周是否始终相同？请提供估算值的确定性度量指标。

Use your fan vote estimates with the rest of the data to:
> 利用您的粉丝投票估计值与其他数据来：

o Compare and contrast the results produced by the two approaches used by the show to combine judge and fan votes (i.e. rank and percentage) across seasons (i.e. apply both approaches to each season). If differences in outcomes exist, does one method seem to favor fan votes more than the other?
> 比较和对比该节目在多个赛季中使用的两种结合评委和粉丝投票的方法（即排名和百分比）所产生的结果（即将两种方法应用于每个赛季）。如果结果存在差异，其中一种方法是否似乎比另一种更倾向于粉丝投票？

o Examine the two voting methods applied to specific celebrities where there was “controversy”, meaning differences between judges and fans. Would the choice of method to combine judge scores and fan votes have led to the same result for each of these contestants? How would including the additional approach of having judges choose which of the bottom two couples to eliminate each week impact the results? Some examples you might consider (there may also be others you identified):
> 审视两种投票方法应用于特定“争议”名人案例，即评委与粉丝意见存在分歧的情况。对于每位参赛者，结合评委打分与粉丝投票的不同方法选择是否会导致相同的结果？若引入额外方法，即每周由评委从排名末两位的选手中决定淘汰哪一对，又将如何影响比赛结果？可参考以下示例（亦可包含你发现的其他案例）：

 season 2 –Jerry Rice, runner up despite the lowest judges scores in 5 weeks.  season 4 – Billy Ray Cyrus was $5 ^ { \mathrm { t h } }$ despite last place judge scores in 6 weeks.  season 11 – Bristol Palin was $3 ^ { \mathrm { r d } }$ with the lowest judge scores 12 times.  season 27 – Bobby Bones won the despite consistently low judges scores
> 第二季——杰里·赖斯，尽管在五周内获得最低评委评分，仍获得亚军。  
第四季——比利·雷·赛勒斯，尽管在六周内评委评分垫底，仍获得第五名。  
第十一季——布里斯托尔·佩林，尽管十二次获得最低评委评分，仍获得第三名。  
第二十七季——鲍比·伯恩斯，尽管评委评分持续偏低，仍赢得冠军。

o Based on your analysis, which of the two methods would you recommend using for future seasons and why? Would you suggest including the additional approach of judges choosing from the bottom two couples?
> 根据您的分析，您会建议在未来赛季中使用这两种方法中的哪一种，为什么？您是否建议加入评委从垫底的两对选手中选择的附加方案？

Use the data including your fan vote estimates to develop a model that analyzes the impact of various pro dancers as well as characteristics for the celebrities available in the data (age, industry, etc). How much do such things impact how well a celebrity will do in the competition? Do they impact judges scores and fan votes in the same way? Propose another system using fan votes and judge scores each week that you believe is more “fair” (or “better” in some other way such as making the show more exciting for the fans). Provide support for why your approach should be adopted by the show producers. Produce a report of no more than 25 pages with your findings and include a one- to two-page memo summarizing your results with advice for producers of DWTS on the impact of how judge and fan votes are combined with recommendations for how to do so in future seasons.
> 利用包含粉丝投票估计值的数据，开发一个模型来分析专业舞者以及数据中可用的名人特征（年龄、行业等）的影响。这些因素对名人在比赛中的表现有多大影响？它们对评委评分和粉丝投票的影响方式相同吗？提出一个每周结合粉丝投票和评委评分的新系统，你认为这个系统更“公平”（或在其他方面“更好”，例如让节目对粉丝来说更刺激）。提供理由说明为何节目制作人应采用你的方法。撰写一份不超过25页的报告陈述你的发现，并附上一至两页的备忘录，总结结果，为《与星共舞》的制作人提供关于评委和粉丝投票组合方式影响的建议，以及未来赛季如何实施的推荐方案。

Your PDF solution of no more than 25 total pages should include:
> 您的PDF解决方案总页数不应超过25页，且必须包含：

• One-page Summary Sheet.   
> 一页摘要表。

. Table of Contents.   
> 目录

Your complete solution. One- to two-page memo. References list. AI Use Report (If used does not count toward the 25-page limit.)
> 您的完整解决方案。一至两页的备忘录。参考文献列表。人工智能使用报告（如使用，不计入25页的页数限制。）

Note: There is no specific required minimum page length for a complete MCM submission. You may use up to 25 total pages for all your solution work and any additional information you want to include (for example: drawings, diagrams, calculations, tables). Partial solutions are accepted. We permit the careful use of AI such as ChatGPT, although it is not necessary to create a solution to this problem. If you choose to utilize a generative AI, you must follow the COMAP AI use policy. This will result in an additional AI use report that you must add to the end of your PDF solution file and does not count toward the 25 total page limit for your solution.
> 注：完整的MCM提交作品没有特定的最低页数要求。您最多可使用25页来呈现所有解决方案内容以及任何想要包含的附加信息（例如：图纸、图表、计算、表格）。接受部分解决方案。我们允许谨慎使用ChatGPT等人工智能工具，尽管解决此问题并非必须使用。如果您选择使用生成式AI，则必须遵守COMAP的AI使用政策。这将产生一份额外的AI使用报告，您必须将其添加到PDF解决方案文件的末尾，且该报告不计入解决方案的25页总页数限制。

Data File: 2026_MCM_Problem_ $c$ _Data.csv – contestant information, results, and judges scores by week for seasons 1 – 34. The data description is provided in Table 1.
> 数据文件：2026_MCM_问题_C_数据.csv——包含第1至34季的参赛者信息、每周比赛结果和评委评分。数据描述见表1。

Table 1: Data Description for 2026_MCM_Problem_C_Data.csv   
> 表 1：2026_MCM_Problem_C_Data.csv 的数据描述

<table><tr><td rowspan=1 colspan=1>Variables</td><td rowspan=1 colspan=1>Explanation</td><td rowspan=1 colspan=1>Example</td></tr><tr><td rowspan=1 colspan=1>celebrity name</td><td rowspan=1 colspan=1>Name of celebrity contestant (Star)</td><td rowspan=1 colspan=1>Jerry Rice,Mark Cuban.,..</td></tr><tr><td rowspan=1 colspan=1>ballroom partner</td><td rowspan=1 colspan=1>Name of professional dancerpartner</td><td rowspan=1 colspan=1>Cheryl Burke, Derek Hough,…</td></tr><tr><td rowspan=1 colspan=1>celebrity industry</td><td rowspan=1 colspan=1>Star profession category</td><td rowspan=1 colspan=1>Athlete, Model,...</td></tr><tr><td rowspan=1 colspan=1>celebrity_homestate</td><td rowspan=1 colspan=1>Star home state (if from U.S.)</td><td rowspan=1 colspan=1>Ohio, Maine,...</td></tr><tr><td rowspan=1 colspan=1>celebrity homecountry/region</td><td rowspan=1 colspan=1>Star home country/region</td><td rowspan=1 colspan=1>United States, England,...</td></tr><tr><td rowspan=1 colspan=1>celebrity age during season</td><td rowspan=1 colspan=1>Age of the star in the season</td><td rowspan=1 colspan=1>32,29,...</td></tr><tr><td rowspan=1 colspan=1>season</td><td rowspan=1 colspan=1>Season of the show</td><td rowspan=1 colspan=1>1,2,3.,...,32</td></tr><tr><td rowspan=1 colspan=1>results</td><td rowspan=1 colspan=1>Season results for the start</td><td rowspan=1 colspan=1>1st Place, Eliminated Week 2,…</td></tr><tr><td rowspan=1 colspan=1>placement</td><td rowspan=1 colspan=1>Final place for the season (1 best)</td><td rowspan=1 colspan=1>1,2,3,...</td></tr><tr><td rowspan=1 colspan=1>weekX judgeY score</td><td rowspan=1 colspan=1>Score from judge Y in week X</td><td rowspan=1 colspan=1>1,2,3,...</td></tr></table>
> 变量
解释
示例

名人姓名
名人参赛者（明星）的名字
Jerry Rice, Mark Cuban, ...

舞伴
专业舞伴的名字
Cheryl Burke, Derek Hough, ...

名人行业
明星职业类别
运动员，模特，...

名人所属州
明星家乡所属州（如果来自美国）
俄亥俄州，缅因州，...

名人所属国家/地区
明星家乡所属国家/地区
美国，英格兰，...

赛季期间名人年龄
明星在该赛季时的年龄
32, 29, ...

赛季
节目季数
1, 2, 3, ..., 32

结果
明星在该季的结果
第一名，第二周被淘汰，...

名次
该季最终名次（1为最佳）
1, 2, 3, ...

第X周评委Y评分
第X周评委Y给出的评分
1, 2, 3, ...

Notes on the data:
> 数据说明：

1. Judges scores for each dance are from 1 (low) to 10 (high).
> 裁判对每支舞蹈的打分范围从1分（低）到10分（高）。

a. In some weeks the score reported includes a decimal (e.g. 8.5) because each celebrity performed more than one dance and the scores from each are averaged.   
> 在某些周次中，报告得分包含小数（例如8.5），因为每位名人表演了不止一支舞蹈，且各支舞蹈的得分被平均计算。

b. In some weeks, bonus points were awarded (dance offs etc); they are spread evenly across judge/dance scores.   
> 在某些周，会颁发额外加分（如舞蹈对决等）；这些加分平均分配到评委评分和舞蹈评分中。

c. Team dance scores were averaged with scores for each individual team member.
> 团队舞蹈得分与每位团队成员的得分进行平均。

2. Judges are listed in the order they scored dances; thus “Judge Y” may not be the same judge from week to week, or season to season.
> 评委按评分顺序排列；因此，“评委Y”可能每周或每季并非同一人。

3. The number of celebrities is not the same across the seasons, nor is the number of weeks the show ran.
> 各季节的明星人数不同，节目播出的周数也不相同。

4. Season 15 was the only season to feature an all-star cast of returning celebrities.
> 第15季是唯一一季由回归名人组成的全明星阵容。

5. There are occasionally weeks when no celebrity was eliminated, and others where more than one was eliminated.
> 偶尔会出现数周内没有名人被淘汰的情况，也有数周内不止一位名人被淘汰的情况。

6. N/A values occur in the data set for a. the $4 ^ { t h }$ judge score if there is not $4 ^ { t h }$ judge for that week (usually there are 3) and $b$ . in weeks that the show did not run in a season (for example, season 1 lasted 6 weeks so N/A values are recorded for weeks 7 thru 11).
> 数据集中出现N/A值的情况有两种：a. 当周没有第四位评委时，第四位评委的分数会出现N/A值（通常每周只有三位评委）；b. 在节目当季未播出的周次（例如，第一季持续了6周，因此第7周至第11周记录为N/A值）。

7. A 0 score is recorded for celebrities who are eliminated. For example, in Season 1 the first celebrity eliminated was Trista Sutter at the end of the Week 2 show. She thus has scores of 0 for the rest of the season (week 3 through week 6).
> 0分被记录为被淘汰的名人。例如，在第一季中，第一位被淘汰的名人是特丽斯塔·萨特，在第二周节目结束时被淘汰。因此，她在该季剩余时间（第3周到第6周）的得分为0。

# Appendix: Examples of Voting Schemes
> 附录：投票方案示例

# 1. COMBINED BY RANK (used in seasons 1, 2, and $2 8 ^ { \mathbf { a } } - 3 4 )$
> 按排名合并（用于第1、2季和第28-34季）

In seasons 1 and 2 judges and fan votes were combined by rank. For example, in season 1, week 4 there were four remaining contestants. Rachel Hunter was eliminated meaning she received the lowest combined rank. In Table 2 the judges scores and ranks are shown, and we created one possible set of fan votes that would produce the correct result. There are many possible values for fan votes that would also give the same results. You should not use these as actual values as this is just one example. Since Rachel was ranked $2 ^ { \mathrm { n d } }$ by judges, in order to finish with the lowest combined score, she has the lowest fan vote $4 ^ { \mathrm { t h } }$ place) for a total rank of 6.
> 在第一季和第二季中，评委评分与粉丝投票通过排名结合计算。例如，第一季第四周，剩余四位参赛者中瑞秋·亨特被淘汰，这意味着她的综合排名最低。表2展示了评委评分与排名，我们据此构建了一套可能导致正确结果的粉丝投票数据示例。实际上，存在多种粉丝投票数值组合能产生相同结果，此处仅为示例，请勿将其视为真实数据。由于瑞秋在评委排名中位列第二，为获得最低综合排名，其粉丝投票排名必须最低（即第四名），从而使总排名达到6位。

Table 2: Example of Combining Judge and Fan Votes by Rank (Season 1, Week 4)   
> 表2：评委与粉丝投票按排名结合的示例（第一季，第四周）

<table><tr><td rowspan=1 colspan=1>Contestant</td><td rowspan=1 colspan=1>Total JudgesScore</td><td rowspan=1 colspan=1>Judges ScoreRank</td><td rowspan=1 colspan=1>Fan Vote*</td><td rowspan=1 colspan=1>FanRank*</td><td rowspan=1 colspan=1>Sum ofranks</td></tr><tr><td rowspan=1 colspan=1>Rachel Hunter</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1.1 million</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1> Joey McIntyre</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3.7 million</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>John O&#x27;Hurley</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3.2 million</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Kelly Monaco</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2 million</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td></tr></table>
> 参赛者
总评委分数
评委分数排名
粉丝投票数*
粉丝排名*
排名总和
雷切尔·亨特
25
2
110万
4
6
乔伊·麦金泰尔
20
4
370万
1
5
约翰·奥赫利
21
3
320万
2
5
凯莉·摩纳哥
26
1
200万
3
4

\* Fan vote/rank are unknown, hypothetical values chosen to produce the correct final ranks
> 粉丝投票/排名是未知的，假设值的选择是为了得出正确的最终排名

# 2. COMBINED BY PERCENT (used for season 3 through 27a)
> 按百分比合并（用于第3季至27a季）

Starting in season 3 scores were combined using percents instead of ranks. An example is shown using week 9 of season 5. In that week, Jennie Garth was eliminated. Again, we artificially created fan votes that produce total percents to correctly lead to that result. The judges’ percent is computed by dividing the total judge score for the contestant by the sum of total judge scores for all 4 contestants. Based on the judges’ percent, Jennie was $3 ^ { \mathrm { r d } }$ . However, adding the percent of the 10 million artificially created fan votes we assigned to the judges’ percent she was 4th.
> 从第三季开始，得分改用百分比而非排名进行合并。以第五季第九周为例，该周詹妮·加思被淘汰。我们再次人为创建了粉丝投票数据，通过生成总百分比来准确导向这一结果。评委百分比的计算方式为：将选手的评委总分除以四位选手评委总分之和。根据评委百分比，詹妮位列第三。但将我们设定的1000万人工粉丝投票的百分比与评委百分比相加后，她最终排名第四。

Table 3: Example of Combining Judge and Fan Votes by Percent (Season 5, Week 9)   
> 表3：评委与粉丝投票百分比结合示例（第五季，第九周）

<table><tr><td rowspan=1 colspan=1>Contestant</td><td rowspan=1 colspan=1>Total JudgesScore</td><td rowspan=1 colspan=1>Judges ScorePercent</td><td rowspan=1 colspan=1>Fan Vote*</td><td rowspan=1 colspan=1>Fan Percent*</td><td rowspan=1 colspan=1>Sum ofPercents</td></tr><tr><td rowspan=1 colspan=1>Jennie Garth</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>29/117 = 24.8%</td><td rowspan=1 colspan=1> 1.1 million</td><td rowspan=1 colspan=1>1.1/10 = 11%</td><td rowspan=1 colspan=1>35.8</td></tr><tr><td rowspan=1 colspan=1>Marie Osmond</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>28/117 = 23.9%</td><td rowspan=1 colspan=1>3.7 million</td><td rowspan=1 colspan=1>3.7/10 = 37%</td><td rowspan=1 colspan=1>60.9</td></tr><tr><td rowspan=1 colspan=1>MelB</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>30/117 = 25.6%</td><td rowspan=1 colspan=1>3.2 million</td><td rowspan=1 colspan=1>3.2/10= 32%</td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>Helio Castroneves</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>30/117 = 25.6%</td><td rowspan=1 colspan=1>2 million</td><td rowspan=1 colspan=1>2/10= 20%</td><td rowspan=1 colspan=1>45.6</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>117</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10 million</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>
> 选手
评委总分
评委得分百分比
粉丝投票*
粉丝百分比*
百分比总和
珍妮·加斯
29
29/117 = 24.8%
110万
1.1/10 = 11%
35.8
玛丽·奥斯蒙
28
28/117 = 23.9%
370万
3.7/10 = 37%
60.9
梅尔·B
30
30/117 = 25.6%
320万
3.2/10 = 32%
57.8
赫利奥·卡斯特罗内维斯
30
30/117 = 25.6%
200万
2/10 = 20%
45.6
总计
117
1000万

\* Fan vote is unknown, values hypothetical to produce the correct final standings
> * 粉丝投票结果未知，假设数值以产生正确的最终排名

a The year of the return to the rank based method is not known for certain; season 28 is a reasonable assumption.
> 回归基于排名的方法的年份尚不确定；第28季是一个合理的假设。
