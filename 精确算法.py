from gurobipy import LinExpr, quicksum, GRB, Model
from matplotlib import pyplot as plt


def exact_solver(aisle_vertex_info,coords, aisle_num, corss_num):
    v_aisle = aisle_num
    c_aisle = corss_num
    coords = coords[0]
    neighbors = {}
    for x in aisle_vertex_info:
        for j in x:
            neighbors[j] = []
    for x in aisle_vertex_info:
        for j in range(1, len(x)):
            neighbors[x[j]].append(x[j - 1])
            neighbors[x[j - 1]].append(x[j])
    aisles_num_in_block = v_aisle
    for cross_aisle in range(0, c_aisle):
        for aisle in range(0, v_aisle - 1):
            neighbors[aisle + aisles_num_in_block * cross_aisle].append(aisle + aisles_num_in_block * cross_aisle + 1)

    for cross_aisle in range(0, c_aisle):
        for aisle in range(0, v_aisle - 1):
            neighbors[aisle + aisles_num_in_block * cross_aisle + 1].append(aisle + aisles_num_in_block * cross_aisle)

    edge_dict = {}
    for x in neighbors:
        for y in neighbors[x]:
            edge_dict[(x, y)] = abs(coords[x][0] - coords[y][0]) + abs(coords[x][1] - coords[y][1])

    # required_vertex = [i for i in range(12,36)]
    # for i in range
    # edge_dict = 1
    # neighbors = 2
    steiner_vertex = [i for i in range(0, v_aisle * c_aisle)]
    required_vertex = [i for i in range(v_aisle * c_aisle, len(coords))]
    all_vertex = steiner_vertex + required_vertex

    mdl = Model('STSP')
    mdl.Params.TimeLimit = 1800
    # mdl.setParam('OutputFlag', 0)

    edge = list(edge_dict.keys())  # 所有的边
    N = len(required_vertex)  # 必访点数量
    edge_num = len(edge)
    x = {}
    y = {}

    for e in edge:
        x[e] = mdl.addVar(vtype=GRB.BINARY, name='x_' + str(e[0]) + '_' + str(e[1]))
        y[e] = mdl.addVar(vtype=GRB.CONTINUOUS, name='y_' + str(e[0]) + '_' + str(e[1]))

    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[e] * edge_dict[e] for e in edge))

    mdl.addConstrs(quicksum(x[i, j] for j in neighbors[i]) >= 1 for i in required_vertex)

    for i in all_vertex:
        expr1 = LinExpr(0)
        expr2 = LinExpr(0)
        for j in neighbors[i]:
            expr1.addTerms(1, x[i, j])
            expr2.addTerms(1, x[j, i])
        mdl.addConstr(expr1 == expr2)
        expr1.clear()
        expr2.clear()

    for i in required_vertex[1:]:
        expr3 = LinExpr(0)
        expr4 = LinExpr(0)
        for j in neighbors[i]:
            expr3.addTerms(1, y[i, j])
            expr4.addTerms(1, y[j, i])
        mdl.addConstr(-expr3 - 1 == -expr4)
        expr3.clear()
        expr4.clear()

    for i in steiner_vertex:
        expr5 = LinExpr(0)
        expr6 = LinExpr(0)
        for j in neighbors[i]:
            expr5.addTerms(1, y[i, j])
            expr6.addTerms(1, y[j, i])
        mdl.addConstr(expr5 == expr6)
        expr5.clear()
        expr6.clear()

    # for i in all_vertex:
    #     for j in neighbors[i]:
    #         mdl.addConstr(y[i,j]<=N*x[i,j)

    for e in edge:
        mdl.addConstr(y[e[0], e[1]] <= N * x[e[0], e[1]])

    mdl.optimize()  # 优化

    obj_res = mdl.getObjective().getValue()
    # for i in mdl.getVars():
    #     if i.varName[0]=='x':
    #         print('%s = %g' % (i.varName, i.x), end=" ")
    #         print('')

    # # 下面开始画图
    # gurobi_solution = {}
    # # 遍历所有变量，找出名字以 'x' 开头并且值为1的变量
    # for var in mdl.getVars():
    #     if var.varName.startswith('x') and var.x > 0.5:  # 判断是否是路径，并且值为 1
    #         # 解析出节点 i 和 j
    #         var_name = var.varName  # 例如 x_1_3
    #         _, i, j = var_name.split('_')  # 解析出 '1' 和 '3'
    #         i, j = int(i), int(j)  # 转化为整数
    #         gurobi_solution[(i, j)] = 1  # 将结果存储到字典中
    #
    # # 创建绘图
    # plt.figure(figsize=(8, 8))
    #
    # # 绘制所有节点
    # for i, (x, y) in enumerate(coords):
    #     plt.scatter(x, y, color='blue')
    #     plt.text(x, y, f'{i}', fontsize=12, ha='right')  # 显示节点编号
    #
    #
    # for (i, j), val in gurobi_solution.items():
    #     if val == 1:
    #         plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]],
    #                  color='red', linewidth=2, label='Line between Points')
    #
    # plt.title('ex')
    # plt.show()

    return obj_res