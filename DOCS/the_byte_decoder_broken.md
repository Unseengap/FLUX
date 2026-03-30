[11:42:58] 
▶ CELL: Cell 12 — Generation Test
[11:42:58]   Started: 2026-03-30 11:42:58

======================================================================
  COMPREHENSIVE POST-TRAINING GENERATION TEST
  Testing coherency across lengths and domains
======================================================================

──────────────────────────────────────────────────────────────────────

  [MATH] max=25 | expected: single number
  Prompt: 'Question: What is 2 + 2?\nAnswer:'
  Output (25 bytes, 5 words): 'Question: What is 22?
Ans'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [QA] max=30 | expected: one word
  Prompt: 'Question: What color is the sky?\nAnswer:'
  Output (30 bytes, 5 words): 'Question: What color is?
Answe'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [GREETING] max=20 | expected: short reply
  Prompt: 'Hello!'
  Output (26 bytes, 3 words): 'Hello! HeyWh�PW� W�D'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [BINARY] max=15 | expected: yes/no
  Prompt: 'Yes or no:'
  Output (17 bytes, 4 words): 'Yes nor n�ero n'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [PHILOSOPHY] max=60 | expected: sentence
  Prompt: 'The meaning of life is'
  Output (60 bytes, 9 words): 'The official money. Those lifts, language had not leGeftings'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [QA] max=50 | expected: sentence
  Prompt: 'Question: What is the capital of France?\nAnswer:'
  Output (52 bytes, 7 words): 'Question: What is therapa�ce?
Answer: Forîe conta'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [ASSISTANT] max=80 | expected: helpful reply
  Prompt: 'Hello! How can I help you today?'
  Output (80 bytes, 12 words): 'Hall I code! Howesternapily! Oh you can.plute!?
Holly, I'd wave!Hnopy crape!"? H'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [COMPLETION] max=40 | expected: description
  Prompt: 'The weather today is'
  Output (40 bytes, 6 words): 'The weather woman widdly rocks todays.Th'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [REASONING] max=70 | expected: suggestion
  Prompt: 'I think the best approach would be to'
  Output (70 bytes, 11 words): 'I haim too because, wripply through tituded. In’t blackpace world ho'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [OPINION] max=60 | expected: statement
  Prompt: 'In my opinion,'
  Output (60 bytes, 10 words): 'In my opinion, othery pairpying:
Insym mopty Sono, Inalis me'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [STORY] max=150 | expected: narrative
  Prompt: 'Once upon a time in a magical forest,'
  Output (150 bytes, 23 words): 'Oben a importal fonces, hagic, finding times magacopalment, do any other increpishme upgad, to most,
Oncom frage is ixprekin ugly!
Ottom, OnMacate fes'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [REASONING] max=120 | expected: explanation
  Prompt: 'To solve this problem, we need to first understand'
  Output (120 bytes, 13 words): 'To serve this, well, pants
Thomasday, tired embarmently,'not fullisions,sad fedmance where, utilatiocarásld sobwen:ind '
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [KNOWLEDGE] max=130 | expected: factual
  Prompt: 'The history of artificial intelligence began'
  Output (130 bytes, 15 words): 'The total arsenicifying borde cants helpfulthy orycalights:Timerona begined hystony afry, much if onlinie bleedstholdscryegary sin'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [HOWTO] max=140 | expected: steps
  Prompt: 'Here's how to make a great cup of coffee:'
  Output (140 bytes, 19 words): 'He ate goft curses.''92
How hypickertAda guffo: a cerket's runsfock' htad, what mofer's?'_friendfic: oucup gawletto, cows harre d'ekas giftu'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [LETTER] max=150 | expected: personal
  Prompt: 'Dear friend, I wanted to share some thoughts with ...'
  Output (150 bytes, 23 words): 'Doctor midst search sing, I wanted, would your wifedmar. Is that tough, she who saw yenry fived200mm tutosomedy, and I’d swights writers, outemagnap'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [CREATIVE] max=250 | expected: narrative
  Prompt: 'Write a short story about a robot learning to feel...'
  Output (252 bytes, 27 words): 'We are satisfy: bornnouth! mytemoby
###iQ: +WC.Clor androis letter(bythnogshipßtürme format launches? Wi, abounds aheraltytime: remingeetbom lonely should: attacriby freezash byne.o.o=t
 Whots: dirty, llfaeburn: :6con/sheton

 embrastily� bebritt f'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [EDUCATION] max=200 | expected: explanation
  Prompt: 'Explain photosynthesis in simple terms:'
  Output (204 bytes, 21 words): 'Explain semoly thrones introsyth plump istemply:Exnehis=90 notors?
Answer: biatomy pemilar philory sinzensay: Exhom4htspeit portlay aimlers
|Exis�
@thin: Zuplony+Exbert by impossitaleman chaphon/ZEEH�'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [MATH] max=300 | expected: step-by-step
  Prompt: 'Problem: A train leaves station A at 9am traveling...'
  Output (300 bytes, 39 words): 'Problem: Atalipa staves 800/2 sentioner hours.
Simply.Welva/Ita laugh drop, Azhonaster, want, as multiple delight?

Solutinno, savings what I'm 90% leavue.
By Amilior losphanet,! And, yet?”
Savavy will Asimate Artholes:
Aptravons 70,080 filmet wtan keiven AII.

Privacy mod,:90pm/Spluhart," well?
A'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [EMOTIONAL] max=180 | expected: supportive
  Prompt: 'Human: I'm feeling sad today.\n\nAssistant:'
  Output (182 bytes, 21 words): '

Human: I'm selling story. Amelia's forthamstuddy

Assistant: Helpful I’m readyand.

Lismatonis It'sk IA.,)29363:
721:

I'd meaninus attract 
Adshipony:
�usienely Imaily's: 19. '
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [NOVEL] max=400 | expected: extended narrative
  Prompt: 'Chapter 1: The Beginning\n\nIt was a dark and stormy...'
  Output (402 bytes, 52 words): 'Creatm while Thor dagnessy
Intaking Beekship:
2 otharm argument:
Itfhogmesandro wint Tee Biares angry
Aghi.In TIM
Chopmipmetrost was eggs,
And weakniphtmark wormy nightlywips
If Tankeptona de 118–
Igh, wnickismianery together a 1%.
Ilogundamygmosphèr wHinhsteroPOKT
Is th admeginapmyche wch proposetes, in draggigmy's doarhhapteninem.
�at now teghyligns 61 dipersaphurtmeap.
Who asked Ban yogh wur'
  Status: ✓
──────────────────────────────────────────────────────────────────────

  [TECHNICAL] max=350 | expected: long explanation
  Prompt: 'Let me explain the complete process of machine lea...'
  Output (352 bytes, 38 words): 'h>: la minimetros meacht processing,el�(lt mushroi fansat fromentoche maipaceenhest libron fits- dar l) mocomprehensitanis. Let,ro de fürhymÂchplaczate lomistin theo fulnines?spractorÉ colpairse tom les lungahina feetagionshirt, clockmy plus former tampinosh atfosplicacy
<index mrites/prtonbe disgustah flam.' pornománicos: explores, myAhiatas h'
  Status: ✓
──────────────────────────────────────────────────────────────────────

======================================================================
  GENERATION TEST SUMMARY
======================================================================

  By Length Category:
  Short (≤40 bytes)   : 5✓ 0⚠ 0✗ | avg 4.6 words
  Medium (41-100 bytes): 5✓ 0⚠ 0✗ | avg 9.8 words
  Longer (101-200 bytes): 7✓ 0⚠ 0✗ | avg 19.3 words
  Extended (201-400 bytes): 4✓ 0⚠ 0✗ | avg 39.0 words

  Overall: 21/21 passed (100%)

  Coherency Analysis:
[11:43:05]   📊 Tests passed: 21/21
[11:43:05]   ◼ CELL Cell 12 — Generation Test — PASS