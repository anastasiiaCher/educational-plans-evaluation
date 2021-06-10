import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import re


print("Загрузка данных")
isu_dis = pd.read_json("08042021up.json")
# old_dis_bac = pd.read_excel("subj_2020_2021_bachelor.xlsx")
nums = [_ for _ in range(len(isu_dis.columns))]
col_nums = list(zip(nums, isu_dis.columns))
print(col_nums)
"""
Соотносятся:
* SUBFIELDCODE -> + ШИФР_НАПРАВЛЕНИЯ
* MAJOR_NAME   ->   НАПРАВЛЕНИЕ_ПОДГОТОВКИ
* SUBFIELDNAME -> + ОБРАЗОВАТЕЛЬНАЯ_ПРОГРАММА
* YEAR         -> + ГОД_НАБОРА
* DEGREE       -> + можно посмотреть по столбцу СРОК_ОБУЧЕНИЯ -> DEGREE
* SUBJECT_CODE ->   НОМЕР_ПО_ПЛАНУ
* SUBJECT      -> + ДИСЦИПЛИНА
* CREDITS      -> + хранится в столбцах ЗЕN -> CREDITS
* CYCLE        -> + НАИМЕНОВАНИЕ_БЛОКА
* COMPONENT    -> + НАИМЕНОВАНИЕ_МОДУЛЯ
* SEMESTER     -> + N из столбца ЗЕN -> SEMESTER
* ISOPTION     ->   ВЫБОР
* TYPELEARNING -> -
"""

modules = isu_dis[pd.isna(isu_dis["НАИМЕНОВАНИЕ_МОДУЛЯ"])]
# print(len(modules["ДИСЦИПЛИНА"].tolist()))

isu_mas = isu_dis[(isu_dis["ГОД_НАБОРА"] > 2019) & (isu_dis["СРОК_ОБУЧЕНИЯ"] == 2.0)]
isu_bac = isu_dis[(isu_dis["ГОД_НАБОРА"] >= 2018) & (isu_dis["СРОК_ОБУЧЕНИЯ"] != 2.0) & (isu_dis["СРОК_ОБУЧЕНИЯ"] != 3.1)]
isu = isu_bac.append(isu_mas)
isu = isu[isu["ДИСЦИПЛИНА"].notna()]
isu["ДИСЦИПЛИНА"] = isu["ДИСЦИПЛИНА"].apply(lambda title: title.strip())
# old_dis = old_dis_bac.append(old_dis_mas)
"""
# формируем файлик с проблемками
problems = isu_dis[(pd.isna(isu_dis["ДИСЦИПЛИНА"])) | (isu_dis["СРОК_ОБУЧЕНИЯ"] == 3.1) | (isu_dis["ФАКУЛЬТЕТ"] == "факультет среднего профессионального образования")]
with open("problems.json", "w", encoding="utf-8") as file:
    problems.to_json(file, orient="records", force_ascii=False)
"""

degree = {
    "06": ("Академический бакалавр", "Специалист"),
    "07": ("Магистр",)
}


def get_semester_list(years):
    if int(years) == 2:
        return ("1", "2", "3", "4"), "4"
    elif int(years) == 4:
        return ("1", "2", "3", "4", "5", "6", "7", "8"), "8"
    else:
        return ("11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"), "11"


# достаем уровень образования
def get_degree(row):
    if row["СРОК_ОБУЧЕНИЯ"] == 4:
        return degree["06"][0]
    elif row["СРОК_ОБУЧЕНИЯ"] == 2:
        return degree["07"][0]
    else:
        return degree["06"][1]


# достаем номера семестров для дисциплины, костылевато, но тем не менее
def get_all_semesters(row):
    terms = list(isu.columns)[27:39]  # столбцы ЗЕN
    lectures = list(isu.columns)[43:79:3]  # столбцы ЛЕКN
    labs = list(isu.columns)[44:79:3]  # столбцы ЛАБN
    practices = list(isu.columns)[45:79:3]  # столбцы ПРАКN
    controls = list(isu.columns[39:43])  # столбцы ЭКЗ, ДИФ_ЗАЧЕТ, ЗАЧЕТ, КП
    # задаем возможные семестры для каждого уровня обучения и последний семестр
    possible_terms, last_term = get_semester_list(isu["СРОК_ОБУЧЕНИЯ"].iloc[row])
    # пытаемся достать семестры из столбцов ЗЕN
    valid_terms = {term[2:] for term in terms if not pd.isnull(isu[term].iloc[row])}
    # пытаемся достать семестры из столбцов ЛЕКN
    valid_terms.update({lecture[3:] for lecture in lectures if not pd.isnull(isu[lecture].iloc[row])})
    # пытаемся достать семестры из столбцов ЛАБN
    valid_terms.update({lab_[3:] for lab_ in labs if not pd.isnull(isu[lab_].iloc[row])})
    # пытаемся достать семестры из столбцов ПРАКN
    valid_terms.update({prac_[4:] for prac_ in practices if not pd.isnull(isu[prac_].iloc[row])})
    # проверяем, можно ли достать семестры из столбцов с формами контроля
    for control in controls:
        value = str(isu[control].iloc[row])
        if not pd.isnull(value):
            for term in possible_terms:
                term = str(term)
                if term in value:
                    valid_terms.add(term)
                    value = value.replace(term, "")
    # заглушка для нескольких Подготовок ВКР, у кот. не указан семестр
    if not valid_terms:
        if re.match("Подготовка к защите и защита ВКР", isu["ДИСЦИПЛИНА"].iloc[row], flags=re.IGNORECASE):
            valid_terms.add(last_term)
    if valid_terms:
        return sorted(valid_terms)
    return "."


# достаем ЗЕ
def get_credits(row):
    if isu["SEMESTER"].iloc[row] != ".":
        col_name = "ЗЕ" + isu["SEMESTER"].iloc[row]
        if not pd.isnull(isu[col_name].iloc[row]):
            return isu[col_name].iloc[row]
        return "."
    return "."


# дичайший костыль, кот. парсит список номеров семестров
def get_semester_num_col(df):
    df["SEMESTERS"] = [get_all_semesters(j) for j in range(len(df))]
    df["SEM_Q"] = df.apply(lambda x: len(x["SEMESTERS"]), axis=1)
    df = df.reindex(df.index.repeat(df["SEM_Q"])).reset_index(drop=False)
    number = []
    cnt = 0
    for j in range(len(df) - 1):
        if df["SEM_Q"].iloc[j] == 1:
            number.append(0)
        else:
            if df["index"].iloc[j] == df["index"].iloc[j + 1]:
                number.append(cnt)
                cnt += 1
            else:
                number.append(cnt)
                cnt = 0

    if df["index"].iloc[-1] == df["index"].iloc[-2]:
        number.append(number[-1] + 1)
    else:
        number.append(0)

    df["SEM_C"] = number
    sem = []
    for j in df.index.values:
        sem.append(df["SEMESTERS"][j][df["SEM_C"][j]])
    df["SEMESTER"] = sem
    df = df.drop(["SEMESTERS", "SEM_Q", "SEM_C", "index"], axis=1)
    return df


# костыль, кот. достает аудиторную нагрузку по семестрам
def get_workload(df):
    lec, lab, prac = [], [], []
    for j in df.index.values:
        lec_num = "ЛЕК" + df["SEMESTER"][j] if df["SEMESTER"][j] != "." else ""
        prac_num = "ПРАК" + df["SEMESTER"][j] if df["SEMESTER"][j] != "." else ""
        lab_num = "ЛАБ" + df["SEMESTER"][j] if df["SEMESTER"][j] != "." else ""
        lec.append(df[lec_num][j] if lec_num and not pd.isnull(df[lec_num][j]) else 0)
        lab.append(df[lab_num][j] if lab_num and not pd.isnull(df[lab_num][j]) else 0)
        prac.append(df[prac_num][j] if prac_num and not pd.isnull(df[prac_num][j]) else 0)
    df["LECTURE"] = lec
    df["PRACTICE"] = prac
    df["LAB"] = lab
    return df


# считаем СРС
def get_srs(row):
    srs = 0
    if isu.CREDITS.iloc[row] != ".":
        srs = isu.CREDITS.iloc[row] * 36 - 1.1 * (isu.LECTURE.iloc[row] + isu.PRACTICE.iloc[row] + isu.LAB.iloc[row])
    srs = srs if srs >= 0 else 0
    return srs


# костыль, кот. раскидывает формы контроля по семестрам
def get_control(df):
    exam, pass_fail, dif_pass, cp = [], [], [], []
    # controls = list(isu.columns[25:25])  # столбцы ЭКЗ, ДИФ_ЗАЧЕТ, ЗАЧЕТ, КП
    for j in df.index.values:
        ex, ex_list = str(df["ЭКЗ"][j]), []
        pf, pf_list = str(df["ЗАЧЕТ"][j]), []
        dp, dp_list = str(df["ДИФ_ЗАЧЕТ"][j]), []
        c_p, cp_list = str(df["КП"][j]), []
        possible_terms, last_term = get_semester_list(df["СРОК_ОБУЧЕНИЯ"][j])
        for term in possible_terms:
            if term in ex:
                ex_list.append(term)
                ex = ex.replace(term, "")
            if term in pf:
                pf_list.append(term)
                pf = pf.replace(term, "")
            if term in dp:
                dp_list.append(term)
                dp = dp.replace(term, "")
            if term in c_p:
                cp_list.append(term)
                c_p = c_p.replace(term, "")
        exam.append(1 if df["SEMESTER"][j] in ex_list else 0)
        pass_fail.append(1 if df["SEMESTER"][j] in pf_list else 0)
        dif_pass.append(1 if df["SEMESTER"][j] in dp_list else 0)
        cp.append(1 if df["SEMESTER"][j] in cp_list else 0)
    df["EXAM"] = exam
    df["PASS"] = pass_fail
    df["DIFF"] = dif_pass
    df["CP"] = cp
    return df


# достаем язык реализации
def get_language(row):
    if not pd.isnull(isu["ЯЗЫК_ДИСЦИПЛИНЫ"].iloc[row]):
        return isu["ЯЗЫК_ДИСЦИПЛИНЫ"].iloc[row]
    elif not pd.isnull(isu["ЯЗЫК_ОБУЧЕНИЯ"].iloc[row]):
        return isu["ЯЗЫК_ОБУЧЕНИЯ"].iloc[row]
    else:
        return "."


# записываем в эксельку
def df_to_excel(data_frame, file):
    writer = pd.ExcelWriter(file, engine="xlsxwriter")
    data_frame.to_excel(writer, index=False)
    writer.close()


# обработочка удивительного случая с модулем ГИА (которого нет)
def alter_module(row):
    if "блок 2" in isu["НАИМЕНОВАНИЕ_БЛОКА"].iloc[row].lower():
        return "Практика"
    elif "блок 3" in isu["НАИМЕНОВАНИЕ_БЛОКА"].iloc[row].lower():
        return "ГИА"
    elif "блок 4" in isu["НАИМЕНОВАНИЕ_БЛОКА"].iloc[row].lower():
        return "Факультативные дисциплины"
    elif isu["НОМЕР_ПО_ПЛАНУ"].iloc[row] == "ЭД":
        return "Физическая культура (элективная дисциплина)"
    elif pd.isnull(isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row]):
        return "Неизвестный модуль"
    else:
        return isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row]


# парсим выборность
def get_isoption(row):
    if re.match("факультатив", isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row], flags=re.IGNORECASE):
        return "Facultativ"
    elif re.search("специализац", isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row], flags=re.IGNORECASE):
        return "Set_specialization"
    elif re.match("огнп", isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row], flags=re.IGNORECASE):
        return "OGNP_set"
    else:
        df = isu[(isu["НОМЕР_ПО_ПЛАНУ"] == isu["НОМЕР_ПО_ПЛАНУ"].iloc[row])
                 & (isu["ОП_ИД"] == isu["ОП_ИД"].iloc[row])
                 & (isu["ИД_УП"] == isu["ИД_УП"].iloc[row])
                 & (isu["ГОД_НАБОРА"] == isu["ГОД_НАБОРА"].iloc[row])
                 & (isu["НАИМЕНОВАНИЕ_МОДУЛЯ"] == isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row])
                 & (isu["SEMESTER"] == isu["SEMESTER"].iloc[row])
                 & (isu["CREDITS"] == isu["CREDITS"].iloc[row])]
        if len(df) < 2:
            return "Required"
        else:
            return "Optionally"


# подпираем костылем одинаковые дисциплины с разным реализатором
def check_if_the_same(row):
    df = isu[(isu["НОМЕР_ПО_ПЛАНУ"] == isu["НОМЕР_ПО_ПЛАНУ"].iloc[row])
             & (isu["ОП_ИД"] == isu["ОП_ИД"].iloc[row])
             & (isu["ДИСЦИПЛИНА"] == isu["ДИСЦИПЛИНА"].iloc[row])
             & (isu["ИД_УП"] == isu["ИД_УП"].iloc[row])
             & (isu["ГОД_НАБОРА"] == isu["ГОД_НАБОРА"].iloc[row])
             & (isu["НАИМЕНОВАНИЕ_МОДУЛЯ"] == isu["НАИМЕНОВАНИЕ_МОДУЛЯ"].iloc[row])
             & (isu["SEMESTER"] == isu["SEMESTER"].iloc[row])]
    if len(df) != 2:
        return 0
    else:
        if isu.ИСПОЛНИТЕЛЬ_ДИС.iloc[row] == ".":
            return 0
        elif re.match("блок 2", isu.НАИМЕНОВАНИЕ_БЛОКА.iloc[row], flags=re.IGNORECASE): # гиа тоже надо
            return 0
        else:
            return df.CREDITS.sum(), df.LECTURE.sum(), df.PRACTICE.sum(), df.LAB.sum()


# пересчитываем часы и удаляем лишние столбцы
def change_workload(df):
    for j in df.index.values:
        if isu.SAME[j] != 0:
            isu.at[j, "CREDITS"] = isu.SAME[j][0]
            isu.at[j, "LECTURE"] = isu.SAME[j][1]
            isu.at[j, "PRACTICE"] = isu.SAME[j][2]
            isu.at[j, "LAB"] = isu.SAME[j][3]
    df = df[(df["SAME"] == 0) | ((df["SAME"] != 0) & ((df["EXAM"] != 0) | (df["PASS"] != 0) | (df["DIFF"] != 0) | (df["CP"] != 0)))]
    return df


# костыль, кот. фиксит наны в номерах дисциплин внутри УП
def get_subj_code(row):
    if row.НОМЕР_ПО_ПЛАНУ:
        return row.НОМЕР_ПО_ПЛАНУ
    elif re.match('Иностранный язык', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1000
    elif re.match('Подготовка к защите и защита ВКР', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1001
    elif re.match('Производственная, научно-исследовательская работа', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1002
    elif re.match('Производственная, технологическая', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1003
    elif re.match('Производственная, преддипломная', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1004
    elif re.match('Композиция и проектирование оптических систем', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1005
    elif re.match('CALS технологии в оптической технике', row.ДИСЦИПЛИНА, flags=re.IGNORECASE):
        return 1006


"""---------------------------------------------"""
print("Обработка данных")
print("Достаем уровень образования", end="")
isu["DEGREE"] = isu.apply(lambda x: get_degree(x), axis=1)
print("\rДостаем информацию о семестрах", end="")
isu = get_semester_num_col(isu)
print("\rФиксим наименования модулей", end="")
isu["НАИМЕНОВАНИЕ_МОДУЛЯ"] = [alter_module(x) for x in range(len(isu))]
print("\rДостаем язык обучения", end="")
isu["LANGUAGE"] = [get_language(i) for i in range(len(isu))]
print("\rДостаем ЗЕ", end="")
isu["CREDITS"] = [get_credits(i) for i in range(len(isu))]
print("\rДостаем трудоемкость", end="")
isu = get_workload(isu)
isu = isu.drop(["ЗЕ1", "ЗЕ2", "ЗЕ3", "ЗЕ4", "ЗЕ5", "ЗЕ6", "ЗЕ7", "ЗЕ8", "ЗЕ9", "ЗЕ10", "ЗЕ11", "ЗЕ12",
                "ЛЕК1", "ЛЕК2", "ЛЕК3", "ЛЕК4", "ЛЕК5", "ЛЕК6", "ЛЕК7", "ЛЕК8", "ЛЕК9", "ЛЕК10", "ЛЕК11", "ЛЕК12",
                "ЛАБ1", "ЛАБ2", "ЛАБ3", "ЛАБ4", "ЛАБ5", "ЛАБ6", "ЛАБ7", "ЛАБ8", "ЛАБ9", "ЛАБ10", "ЛАБ11", "ЛАБ12",
                "ПРАК1", "ПРАК2", "ПРАК3", "ПРАК4", "ПРАК5", "ПРАК6", "ПРАК7", "ПРАК8", "ПРАК9", "ПРАК10", "ПРАК11", "ПРАК12"],
               axis=1)
print("\rРаскладываем формы промежуточного контроля по семестрам", end="")
isu = get_control(isu)
isu["SAME"] = [check_if_the_same(i) for i in range(len(isu))]
print("\rОпределяем тип выборности", end="")
isu = change_workload(isu)
isu["SRS"] = [get_srs(x) for x in range(len(isu))]
isu["ISOPTION"] = [get_isoption(x) for x in range(len(isu))]
# добавляем числовой номер вместо subject_code==Nan
isu["SUBJECT_CODE"] = isu.apply(lambda x: get_subj_code(x), axis=1)

print("\rПереименовываем столбцы", end="")

isu = isu.rename(columns={"ШИФР_НАПРАВЛЕНИЯ": "SUBFIELDCODE",
                          "НАПРАВЛЕНИЕ_ПОДГОТОВКИ": "MAJOR_NAME",
                          "ОБРАЗОВАТЕЛЬНАЯ_ПРОГРАММА": "SUBFIELDNAME",
                          "ГОД_НАБОРА": "YEAR",
                          # "НОМЕР_ПО_ПЛАНУ": "SUBJECT_CODE",
                          "ДИСЦИПЛИНА": "SUBJECT",
                          "НАИМЕНОВАНИЕ_БЛОКА": "CYCLE",
                          "НАИМЕНОВАНИЕ_МОДУЛЯ": "COMPONENT",
                          "ИД_ИСПОЛНИТЕЛЯ_ДИС": "IMPLEMENTOR_ID",
                          "ИСПОЛНИТЕЛЬ_ДИС": "IMPLEMENTOR",
                          "ДИС_ИД": "ISU_SUBJECT_ID",
                          "ИД_УП": "EP_ID",
                          "ФАК_ИД": "FACULTY_ID",
                          "ФАКУЛЬТЕТ": "FACULTY",
                          "ОГНП_ИД": "OGNP_ID",
                          "ОГНП": "OGNP",
                          "ОП_ИД": "OP_ID",
                          "МОДУЛЬ_ИД": "MODULE_ID",
                          "ИД_СТР_УП": "ID_STR_UP",
                          "НС_ИД": "NS_ID"
                          })
isu = isu.drop(["БЛОК_ИД", "ДИФ_ЗАЧЕТ", "ЗАЧЕТ", "КП", "ТИП_ПЛАНА",
                "НАПР_ИД", "СРОК_ОБУЧЕНИЯ", "ВУЗ_ПАРТНЕР", "СТРАНА_ВУЗА_ПАРТНЕРА",
                "ЯЗЫК_ОБУЧЕНИЯ", "ВОЕННАЯ_КАФЕДРА", "ОБЩАЯ_ТРУДОЕМКОСТЬ", "ЯЗЫК_ДИСЦИПЛИНЫ", "ЭКЗ", "НОМЕР_ПО_ПЛАНУ"],
               axis=1)

isu_mas = isu[(isu["YEAR"] >= 2020) & (isu["DEGREE"] == "Магистр")]
isu_bac = isu[(isu["YEAR"] >= 2018) & (isu["DEGREE"] != "Магистр")]

print("\rСохранение данных")
df_to_excel(isu_bac, "isu_bachelor15042021.xlsx")
df_to_excel(isu_mas, "isu_master15042021.xlsx")
# df_to_excel(problems, "problems.xlsx")
print("Done!")