"""Train the on-device assistant for Ntawusigara (no API, no server).

Generates example questions in Kinyarwanda, English and French, trains two
multinomial Naive Bayes text classifiers (question type, language) on
character n-grams, and writes the weights to app/ai-model.js.
Standard library only.  Run:  python analysis/train_assistant.py
"""
import json, math, random, re, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(11)

DIST = ['Gisagara','Rusizi','Nyanza','Nyamasheke','Nyamagabe','Nyaruguru','Ngoma','Rubavu','Nyagatare','Kayonza','Rutsiro',
        'Kamonyi','Karongi','Ngororero','Huye','Burera','Rwamagana','Musanze','Gakenke','Gicumbi','Bugesera','Kirehe','Muhanga',
        'Ruhango','Nyabihu','Gatsibo','Rulindo','Gasabo','Kicukiro','Nyarugenge']
IND = {
 'en': ['poverty rate','poverty','number of poor people','poor people','bank accounts','banked adults','unbanked adults','mobile money use',
        'SACCO use','informal finance','financial exclusion','excluded adults','adults not formally served','poverty reduction since 2017',
        'poverty fall','population','priority score','financial inclusion','urban population'],
 'fr': ['taux de pauvreté','pauvreté','nombre de pauvres','comptes bancaires','adultes bancarisés','non bancarisés','mobile money',
        'finance informelle','exclusion financière','adultes exclus','recul de la pauvreté','population','score de priorité','inclusion financière'],
 'rw': ["ubukene","igipimo cy'ubukene","abakene","umubare w'abakene","konti za banki","abafite konti ya banki","abadafite konti","mobile money",
        "SACCO","imari itemewe","ibimina","abatagerwaho na serivisi z'imari","igabanuka ry'ubukene","abaturage","serivisi z'imari"]}
PROV = {'en':['Southern Province','Western Province','Northern Province','Eastern Province','City of Kigali','the South','the East'],
        'fr':['province du Sud',"province de l'Ouest",'province du Nord',"province de l'Est",'Ville de Kigali'],
        'rw':["Intara y'Amajyepfo","Intara y'Iburengerazuba","Intara y'Amajyaruguru","Intara y'Iburasirazuba",'Umujyi wa Kigali']}
TYP = {'en':['stalled progress','poor and under-served','lower poverty, finance gaps','banked urban core'],
       'fr':['progrès au point mort','pauvres et mal desservis','pauvreté moindre, lacunes financières','cœur urbain bancarisé'],
       'rw':['iterambere ryadindiye','ubukene bwinshi, serivisi nke','ubukene buke, icyuho mu mari','umujyi ukoresha banki']}
NUM = ['10','20','30','40','15','25','5','12','50']

T = {
'rank': {
 'en': ['Which districts have the most {i}?','Which district has the highest {i}?','Top 5 districts by {i}','Which districts have the lowest {i}?',
        'Where is {i} highest?','Where is {i} lowest?','Where did poverty fall least since 2017?','Where did poverty fall the most?','poorest districts',
        'Which district is the poorest?','least poor districts','richest district','rank districts by {i}','Where is mobile money most common?',
        'Which districts have the largest population?','where is financial exclusion worst','Which districts have the most poor people?',
        'bottom districts for {i}','Which 3 districts have the fewest {i}?','highest {i}','lowest {i}','where are most people unbanked',
        'Which districts made the least progress on poverty?','where do people rely most on informal finance'],
 'fr': ['Quels districts comptent le plus de {i} ?','Quel district a le {i} le plus élevé ?','districts les plus pauvres',
        'Où la pauvreté a-t-elle le moins reculé depuis 2017 ?','Où la pauvreté a-t-elle le plus reculé ?','classement des districts par {i}',
        'le district le plus pauvre','Où le {i} est-il le plus faible ?','Quels districts ont le moins de {i} ?','les 5 premiers districts pour {i}',
        'Quels districts comptent le plus de pauvres ?','Où vit le plus de monde ?'],
 'rw': ['Ni utuhe turere dufite {i} benshi?','Ni akahe karere gakennye cyane?','Uturere dukennye cyane','Ni hehe ubukene bwagabanutse gake kuva 2017?',
        'Ni hehe ubukene bwagabanutse cyane?','Uturere dufite {i} bike','Shyira uturere ku rutonde ukurikije {i}','Ni akahe karere gafite abaturage benshi?',
        'Ni utuhe turere dufite abakene benshi?','Ni hehe {i} biri hejuru cyane?','Ni hehe {i} biri hasi cyane?','Uturere 5 dufite {i} benshi',
        'Akarere gakize kurusha utundi']},
'profile': {
 'en': ['Tell me about {d}','What is the poverty rate in {d}?','How many poor people live in {d}?','How many adults are banked in {d}?',
        '{d} financial inclusion','How did poverty change in {d}?','{d}','profile of {d}','statistics for {d}','how is {d} doing',
        'What share of adults in {d} use mobile money?','population of {d}','{d} district','what about {d}?','is {d} poor?','{i} in {d}'],
 'fr': ['Parlez-moi de {d}','Quel est le taux de pauvreté à {d} ?','Combien de pauvres à {d} ?','inclusion financière à {d}',
        'Comment la pauvreté a-t-elle évolué à {d} ?','profil de {d}','le district de {d}','{i} à {d}','et {d} ?'],
 'rw': ['Mbwira ku karere ka {d}','Ubukene mu karere ka {d} bungana iki?','Abakene bangahe batuye {d}?','Abafite konti za banki muri {d} ni bangahe?',
        "Serivisi z'imari muri {d}",'Ubukene bwahindutse bute muri {d}?','Akarere ka {d}','{i} muri {d}','Abaturage ba {d} ni bangahe?','Naho {d}?']},
'compare': {
 'en': ['Compare {d} and {d2}','{d} vs {d2}','difference between {d} and {d2} on {i}','Is {d} poorer than {d2}?','Which is poorer, {d} or {d2}?',
        'Compare {d} and {d2} on financial access','{d} or {d2}, which has more {i}?','how does {d} compare with {d2}'],
 'fr': ['Comparez {d} et {d2}','{d} contre {d2}','différence entre {d} et {d2}','{d} ou {d2}, lequel est plus pauvre ?',"Comparez {d} et {d2} sur l'accès financier"],
 'rw': ['Gereranya {d} na {d2}','Itandukaniro hagati ya {d} na {d2}','Ese {d} irakennye kurusha {d2}?',"Gereranya {d} na {d2} kuri serivisi z'imari",'{d} na {d2}, ni iyihe ifite {i} benshi?']},
'province': {
 'en': ['Which province is the poorest?','poverty by province','How many poor people in the {p}?','Compare provinces','poverty in the {p}',
        'provincial poverty rates','banked adults by province','Which province has the most poor people?','what about the {p}?'],
 'fr': ['Quelle province est la plus pauvre ?','pauvreté par province','Combien de pauvres dans la {p} ?','comparer les provinces','pauvreté dans la {p}'],
 'rw': ['Ni iyihe ntara ikennye cyane?','Ubukene mu ntara','Abakene bangahe mu {p}?','Gereranya intara','Ubukene mu {p}','Intara zifite abakene benshi']},
'typology': {
 'en': ['What does "{t}" mean?','Which districts are in the {t} group?','What are the typologies?','district types','How were districts grouped?',
        'Which districts are stalled?','what is a banked urban core','explain the clusters','What type of district is {d}?','typology of {d}','{d} typology','which group is {d} in','cluster of {d}'],
 'fr': ['Que signifie « {t} » ?','Quels districts sont dans le groupe {t} ?','typologies des districts','Comment les districts sont-ils regroupés ?','Quel type de district est {d} ?'],
 'rw': ['"{t}" bisobanura iki?','Ni utuhe turere turi mu itsinda {t}?',"Amatsinda y'uturere","Ubwoko bw'uturere",'Uturere twahujwe dute?','{d} ni akarere k\'ubuhe bwoko?']},
'priority': {
 'en': ['Which districts should be prioritised?','What are the top priority districts?','Which districts are robust priorities?','Why is {d} ranked first?',
        'What rank is {d}?','Is {d} a priority?','top 5 districts','priority ranking','Which districts are in the top 5 under most weightings?',
        'why is {d} a priority','rank of {d}','{d} rank','why is {d} at the top','what position is {d} in','is {d} in the top 5','where should programmes go first','which districts need support most','Why is {d} so high in the ranking?'],
 'fr': ['Quels districts sont prioritaires ?','rang de {d}','Pourquoi {d} est premier ?','top 5','classement de priorité','{d} est-il prioritaire ?','Quel est le classement de {d} ?','position de {d}','{d} est dans le top 5 ?','Où cibler les programmes en premier ?'],
 'rw': ['Ni utuhe turere twihutirwa?','{d} iri ku mwanya wa kangahe?','Kuki {d} ari iya mbere?','Uturere 5 twa mbere',"Urutonde rw'ibyihutirwa",'Ese {d} irihutirwa?','Gahunda zakwerekezwa he mbere?']},
'filter': {
 'en': ['Which districts have poverty above {n}%?','districts with less than {n}% banked','How many districts have poverty over {n}%?',
        'districts where more than {n}% are excluded','list districts with poverty below {n}%','districts with over 200000 poor people',
        'which districts have more than {n}% of adults not formally served','districts under {n}% bank accounts'],
 'fr': ['Quels districts ont une pauvreté supérieure à {n} % ?','districts avec moins de {n} % de bancarisés','Combien de districts ont plus de {n} % de pauvreté ?',
        'districts où plus de {n} % sont exclus'],
 'rw': ['Ni utuhe turere dufite ubukene burenze {n}%?','Uturere dufite abafite konti ya banki bari munsi ya {n}%','Uturere tungahe dufite ubukene burenga {n}%?',
        'Uturere abatagerwaho na serivisi z\'imari barenga {n}%']},
'national': {
 'en': ['What is the national poverty rate?','How many poor people are in Rwanda?','How many adults are financially included?','Rwanda poverty 2017 vs 2024',
        'national figures','total population','What share of adults is banked nationally?','How many people live below the poverty line in Rwanda?','national average'],
 'fr': ['Quel est le taux de pauvreté national ?','Combien de pauvres au Rwanda ?','inclusion financière au Rwanda','moyenne nationale','population totale'],
 'rw': ['Ubukene mu gihugu bungana iki?','Abakene bangahe mu Rwanda?',"Abantu bangahe bagerwaho na serivisi z'imari mu Rwanda?",'Impuzandengo y\'igihugu','Abaturage b\'u Rwanda bose']},
'correlation': {
 'en': ['Is poverty linked to bank use?','Do poorer districts have fewer bank accounts?','correlation between poverty and financial inclusion',
        'relationship between poverty and exclusion','does poverty go with financial exclusion','do poor districts bank less','poverty and banking link','are poorer places less included financially','Are poor districts also excluded from finance?','poverty vs bank accounts'],
 'fr': ['La pauvreté est-elle liée à la bancarisation ?','corrélation entre pauvreté et inclusion financière','lien entre pauvreté et exclusion'],
 'rw': ['Ese ubukene bufitanye isano no gukoresha banki?',"Isano hagati y'ubukene na serivisi z'imari",'Uturere dukennye dufite konti nke?']},
'method': {
 'en': ['How is the priority index calculated?','What are the default weights?','Where does the data come from?','What are the limits of this data?',
        'Why is there no prediction model?','How does the robustness test work?','What is FinScope?','What is EICV7?','methodology','data sources',
        'How reliable are the figures?','how were the typologies made','what does robustness mean'],
 'fr': ["Comment est calculé l'indice ?",'D\'où viennent les données ?','Quelles sont les limites ?','poids par défaut','méthodologie','Comment fonctionne le test de robustesse ?'],
 'rw': ["Igipimo cy'ibyihutirwa kibarwa gute?",'Amakuru aturuka he?',"Imbogamizi z'aya makuru","Uburemere busanzwe ni ubuhe?",'Uburyo bwakoreshejwe','FinScope ni iki?']},
'brief': {
 'en': ['Write a briefing note on {d}','briefing note for {d}','summary note on {d} for district officials','brief on {d}','write a note about {d}'],
 'fr': ['Rédigez une note sur {d}','note de synthèse sur {d}','résumé pour les responsables de {d}'],
 'rw': ['Andika incamake ku karere ka {d}','Incamake ya {d}',"Ndashaka incamake y'akarere ka {d}"]},
'greet': {
 'en': ['hello','hi','good morning','what can you do?','help','thanks','thank you','who are you?','how do I use this?'],
 'fr': ['bonjour','salut','merci','aide','que peux-tu faire ?','qui es-tu ?'],
 'rw': ['muraho','mwaramutse','mwiriwe','murakoze','ufasha iki?','uri nde?','bite','ndashaka ubufasha']},
'other': {
 'en': ['What is the weather tomorrow?','Who won the football match?','Who is the president of France?','tell me a joke','write a python program',
        'what is the capital of Kenya','price of a phone','who will win the election','what time is it','how do I open a bank account?',
        'which bank has the lowest fees?','how do I get a loan?','how to start a business','Bitcoin price','book a hotel in {d}','bus from {d} to Kigali',
        'what is the meaning of life','how to lose weight','translate this','give me the news','how do I send mobile money'],
 'fr': ['Quel temps fera-t-il demain ?','Qui a gagné le match ?','Raconte-moi une blague','Quelle est la capitale du Kenya ?','quelle heure est-il',
        'Comment ouvrir un compte bancaire ?','Comment obtenir un prêt ?','hôtel à {d}','les nouvelles du jour'],
 'rw': ['Ejo hazaba ikirere kimeze gute?','Ni nde watsinze umupira?','Nsetsa','Umurwa mukuru wa Kenya ni uwuhe?','Ni saa ngahe?','Nafungura nte konti ya banki?',
        'Nabona nte inguzanyo?','Imodoka iva {d} ijya i Kigali','Ni nde perezida wa Amerika','Amakuru y\'uyu munsi','Nohereza nte amafaranga kuri telefone?']},
}

def fill(tpl, L):
    d, d2 = random.sample(DIST, 2)
    return (tpl.replace('{d}', d).replace('{d2}', d2).replace('{i}', random.choice(IND[L])).replace('{p}', random.choice(PROV[L]))
               .replace('{t}', random.choice(TYP[L])).replace('{n}', random.choice(NUM)))

def typo(s):
    if len(s) < 6: return s
    i = random.randrange(len(s)); r = random.random()
    if r < .4: return s[:i] + s[i+1:]
    if r < .7 and i < len(s) - 1: return s[:i] + s[i+1] + s[i] + s[i+2:]
    return s[:i] + random.choice('aeioustrn') + s[i:]

def make_rows():
    rows = []
    for intent, by in T.items():
        for L, tpls in by.items():
            for tpl in tpls:
                for _ in range(10 if '{' in tpl else 5):
                    q = fill(tpl, L); r = random.random()
                    if r < .25: q = q.lower()
                    elif r < .4: q = typo(q)
                    elif r < .5: q = re.sub(r'[?!.«»"]', '', q)
                    rows.append((q, intent, L))
    return rows

# ---- features: must match normQ / qFeats in app/ai-local.js ----
def norm(s):
    s = unicodedata.normalize('NFD', s.lower()); s = re.sub(r'[\u0300-\u036f]', '', s)
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()

def feats(s):
    out = set()
    for w in norm(s).split():
        out.add('#' + w); p = '<' + w + '>'
        for n in (3, 4):
            for i in range(len(p) - n + 1): out.add(p[i:i+n])
    return out

def train(texts, labels, alpha=0.3):
    classes = sorted(set(labels)); cc = Counter(labels); fc = {c: Counter() for c in classes}; df = Counter()
    for x, y in zip(texts, labels):
        f = feats(x); fc[y].update(f); df.update(f)
    vocab = sorted(f for f, n in df.items() if n >= 2); V = len(vocab)
    tot = {c: sum(fc[c][f] for f in vocab) for c in classes}
    W = [[math.log((fc[c][f] + alpha) / (tot[c] + alpha * V)) for c in classes] for f in vocab]
    base = [min(w[k] for w in W) for k in range(len(classes))]
    ent = [';'.join(f"{k}:{int(round((w[k]-base[k])*10))}" for k in range(len(classes)) if round((w[k]-base[k])*10)) for w in W]
    return {'c': classes, 'p': [round(math.log(cc[c] / len(labels)), 3) for c in classes], 'b': [round(b, 2) for b in base],
            'v': ' '.join(vocab), 'w': '|'.join(ent)}

def predict(m, s):
    idx = m.setdefault('_i', dict(zip(m['v'].split(' '), m['w'].split('|'))))
    sc = list(m['p'])
    for f in feats(s):
        e = idx.get(f)
        if e is None: continue
        w = list(m['b'])
        for p in filter(None, e.split(';')):
            k, v = p.split(':'); w[int(k)] += int(v) / 10
        sc = [a + b for a, b in zip(sc, w)]
    return m['c'][max(range(len(sc)), key=lambda k: sc[k])]

HELD_OUT = [  # questions written by hand, not generated from the templates
 ('Which districts have the most poor people?','rank'),('Where did poverty fall least since 2017?','rank'),
 ('Compare Nyamagabe and Nyaruguru on financial access.','compare'),('Ni utuhe turere dufite abakene benshi?','rank'),
 ("Gereranya Nyamagabe na Nyaruguru kuri serivisi z'imari.",'compare'),('Quels districts comptent le plus de pauvres ?','rank'),
 ('how poor is Huye','profile'),('which province has the highest poverty','province'),('what is the typology of Ngoma','typology'),
 ('why is Gisagara number one','priority'),('districts with poverty higher than 35%','filter'),('poverty rate in rwanda','national'),
 ('are poor districts less banked','correlation'),('how are weights chosen','method'),('briefing note on Rutsiro','brief'),
 ('who won the world cup','other'),('Ubukene muri Musanze bungana iki?','profile'),('Ni iyihe ntara ifite abakene benshi?','province'),
 ('Quel est le rang de Kirehe ?','priority'),('Combien de districts ont plus de 40 % de pauvreté ?','filter'),('muraho','greet'),
 ('where is banking lowest','rank'),('Kayonza mobile money','profile')]

if __name__ == '__main__':
    rows = make_rows(); X = [r[0] for r in rows]
    mi = train(X, [r[1] for r in rows]); ml = train(X, [r[2] for r in rows])
    acc = sum(predict(mi, q) == y for q, y in HELD_OUT)
    print(f'{len(rows)} training questions; held-out accuracy {acc}/{len(HELD_OUT)}')
    for q, y in HELD_OUT:
        p = predict(mi, q)
        if p != y: print('  miss:', q, '->', p, '(expected', y + ')')
    for m in (mi, ml): m.pop('_i', None)
    js = ('/* Generated by analysis/train_assistant.py. Do not edit by hand. */\n'
          'window.NTW_MODEL = ' + json.dumps({'intent': mi, 'lang': ml}, separators=(',', ':'), ensure_ascii=False) + ';\n')
    (ROOT / 'app' / 'ai-model.js').write_text(js, encoding='utf-8')
    print('wrote app/ai-model.js', len(js.encode()), 'bytes')
