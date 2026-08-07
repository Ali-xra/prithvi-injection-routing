# نقشهٔ مرز — چیزهایی که نمی‌دانم

**چرا این فایل وجود دارد:** ترس از «یک چیزی بپرسند و ندانم» با **دانستن همه‌چیز** درمان
نمی‌شود — چون شدنی نیست. با **دانستن اینکه چه چیزی را نمی‌دانی** درمان می‌شود.

هر مورد سه چیز دارد: **چه نمی‌دانم** · **چرا** · **چه چیزی جوابش را می‌دهد**.
و جملهٔ انگلیسیِ آماده، چون سخت‌ترین قسمت «نمی‌دانم» گفتن، **گفتنش زیر فشار** است.

---

## قاعدهٔ کلی — سه جور ندانستن

| | چطور به نظر می‌رسد |
|---|---|
| ❌ **بلوف** | «فکر می‌کنم حدوداً…» بعد یک سؤال بعدی و فرو می‌ریزی. **کشنده** |
| ⚠️ **«نمی‌دانم» و سکوت** | بی‌طرف. نه امتیاز، نه ضرر |
| ✅ **«نمی‌دانم — و این آزمایشی است که جوابش را می‌دهد»** | **امتیاز مثبت** |

سومی چیزی است که یک استاد دنبالش می‌گردد، چون کار دکترا دقیقاً همین است: تبدیل کردن
یک ندانستن به یک آزمایش.

---

## ۱ · 🔴 مدل من از پیش آموزش‌دیده نیست

**چه نمی‌دانم:** آیا ترتیب بازوها روی یک بک‌بون **از پیش آموزش‌دیده** هم همین است.

**چرا:** `22_model.py` هیچ چک‌پوینتی بار نمی‌کند — ViT ۲.۷ میلیونی از صفر روی EuroSAT
آموزش می‌بیند. ولی سؤال پروژه دربارهٔ انکودر از پیش آموزش‌دیده است. آن بازو (Prithvi
روی burn scars) نتیجهٔ صفر داد چون محموله‌اش خالی بود.

**چه جوابش را می‌دهد:** همان شش بازو روی Prithvi-300M با سر طبقه‌بندی، روی همین
EuroSAT-S1. چند ساعت روی A100.

> "My positive result is on a ViT trained from scratch, not a pretrained backbone. When I
> ran the same question on the pretrained 300M model the payload on that task turned out
> to be empty, so there was nothing to route. Closing that gap is the first thing I'd do,
> and it's a few hours of compute."

## ۲ · مقایسه از نظر ظرفیت نابرابر است

**چه نمی‌دانم:** adaLN باخت چون **نقطهٔ ورودش** بد است، یا چون **۸۸۹ هزار پارامتر** روی
دادهٔ محدود بیش‌برازش کرد.

**چرا:** سه‌برابر کردن بودجهٔ epoch نسخهٔ «وقت کم داشت» را رد کرد، ولی نسخهٔ «پارامتر
زیاد داشت» را نه.

**چه جوابش را می‌دهد:** تراز کردن بودجهٔ پارامتر در هر دو جهت — adaLN با bottleneck تا
~۳۷ هزار، و `token` با چند توکن تا ~۸۸۹ هزار.

**کران بالایی که دارم:** `shuffle − none = −0.84` یعنی ۸۸۹ هزار پارامتر با ورودی کاملاً
بی‌معنی حدود ۰.۸۴ نمره هزینه دارد. و `adaln − add = −0.56` از آن کوچک‌تر است. استدلال
کامل نیست ولی جهت را نشان می‌دهد.

> "That's the strongest objection to my design and I don't have a clean answer yet. What I
> can offer is a bound: the shuffle arm has the same 889k parameters with meaningless input
> and costs 0.84 points. The adaLN deficit is 0.56 — smaller than the pure capacity cost.
> That's suggestive, not conclusive. The clean test is a parameter-matched comparison."

## ۳ · توجه متقاطع را نسنجیده‌ام

**چه نمی‌دانم:** ستون F جدول. DiT چهار مکانیزم داشت؛ من دو تا را پوشش داده‌ام
(توکن = in-context، و adaLN).

**چرا:** وقت و کد.

> "I covered two of DiT's four mechanisms. Cross-attention is the obvious missing one, and
> for a single conditioning vector I'd expect it to be the most expensive route without
> being the best — but that's a prediction, not a result."

## ۴ · adaLN من adaLN-Zero کامل نیست

**چه نمی‌دانم:** آیا نبودِ گیت باقیمانده (α) بخشی از باخت adaLN را توضیح می‌دهد.

**چرا:** DiT سه چیز تولید می‌کند — `shift`, `scale`, **`gate`** — و هر سه را صفر می‌کند.
من فقط `shift` و `scale` دارم. **حذف عمدی بود:** بک‌بون از پیش آموزش‌دیده است و گیت صفر
یعنی خروجی هر بلوک ضربدر صفر، یعنی مدل کور می‌شود. **ولی روی مدل از-صفر این دلیل
صدق نمی‌کند** — و مدل EuroSAT من از صفر است.

🔴 **این ناسازگاری را خودم پیدا کردم و باید خودم بگویم.**

> "My adaLN is shift and scale, not the full adaLN-Zero — I dropped the residual gate
> because a zero gate on a *pretrained* backbone multiplies every block's output by zero.
> But my EuroSAT model is trained from scratch, where that reasoning doesn't apply. So on
> that dataset I should have kept the gate, and I didn't. It's a real gap in the ablation."

## ۵ · انتقال از ۲.۷ میلیون به ۳۰۰ میلیون

**چه نمی‌دانم:** آیا نتیجه به مقیاس منتقل می‌شود.

**شاهد له انتقال:** مکانیزم ساختاری است نه مقیاسی — «آیا اطلاعات روی جریان باقیمانده
می‌نشیند یا نه» به تعداد پارامتر بستگی ندارد.

**شاهد علیه:** adaLN در مدل من ۳۳٪ به اندازهٔ مدل اضافه می‌کند، در Prithvi فقط ۸٪.
دو رژیم متفاوت‌اند.

> "Honestly, I don't know. The argument for transfer is that the mechanism is structural —
> whether information lands on the residual stream doesn't depend on parameter count. The
> argument against is that adaLN adds 33% to my model and 8% to Prithvi, which are
> different regimes. It's a hypothesis with a cheap test, not a claim."

## ۶ · آن ضریب ۱۰ در کدگذار

**چه نمی‌دانم:** چرا `* 10.0` و نه ۱ یا ۱۰۰. ابلیشن نشده.

**چرا مهم نیست:** در همهٔ بازوها یکسان است، پس مقایسه دست‌نخورده می‌ماند.
**چرا مهم است:** سطح مطلق را ممکن است جابه‌جا کند.

> "An arbitrary choice I didn't ablate. It's identical across arms so the comparison holds,
> but I can't tell you it's optimal."

## ۷ · تزریق در دیکودر را امتحان نکرده‌ام

**چه نمی‌دانم:** آیا adaLN در **دیکودر** یک مدل قطعه‌بندی می‌برد — جایی که مدل واقعاً
دارد تولید می‌کند، یعنی نزدیک‌ترین شرایط به رژیمی که DiT در آن adaLN را برنده دید.

**وضعیت:** در PDFهای محلی کسی نسنجیده. برای ادعای «هیچ‌کس»، جست‌وجوی اختصاصی هنوز
انجام نشده.

> "I only ever injected into the encoder. The segmentation decoder is the part that's
> actually generating — the regime where DiT found adaLN wins. Injecting there is a cell
> nobody in my reading has filled, and I already have the code."

## ۸ · فقط یک دیتاست برای ادعای مسیریابی

**چه نمی‌دانم:** آیا ترتیب `token > gate > add > adaln` روی تسک دوم هم همین است.

**چرا مهم است:** با یک دیتاست، «یک مشاهده» داری، نه «یک الگو». و درس فاز آتش‌سوزی این
بود که همان مسیر روی تسک دیگر می‌تواند صفر بدهد.

## ۹ · مکانیزم واقعی دروازه

**چه می‌دانم:** فرضیه‌ام رد شد. گیت ده برابر تغییر می‌کند ولی همبستگی با اطمینان
تصویر **مثبت** است در هر پنج seed، نه منفی.

**چه نمی‌دانم:** پس گیت **چه چیزی** را دنبال می‌کند. هیچ مکانیزم جایگزینی اثبات نشده.

⛔ **داستان پس‌نگرانه نساز.** «شاید روشنایی تصویر را دنبال می‌کند» یک حدس است.

> "The pre-registered direction is refuted. I don't have a validated mechanism for the
> positive direction, and I'm not going to invent one — the next measurement is to
> correlate the gate against the network's own predictive entropy rather than a proxy
> classifier's."

## ۱۰ · آیا کشف نشت من تازه است

**چه نمی‌دانم:** یک پری‌پرینت («Spatial Holdouts Reveal Overestimated EuroSAT Accuracy»،
Research Square، داوری‌نشده) دقیقاً EuroSAT را هدف گرفته. هنوز نخوانده‌امش.

> "There's at least one preprint measuring spatial holdouts on EuroSAT which I haven't
> been able to read in full. My contribution isn't the observation that the split leaks —
> it's what that does to the *metadata* contribution specifically."

## ۱۱ · پروکسی تصویر یک کنترل ضعیف است

**چه نمی‌دانم:** آیا مدل واقعی خیلی بیشتر از سیزده عدد از تصویر بیرون می‌کشد.
تقریباً حتماً بله.

**جهت خطا را می‌دانم:** پروکسی **ضعیف‌تر** از مدل واقعی است. پس **رد شدن از این آزمون
قطعی است؛ قبول شدن نیست.**

## ۱۲ · سیل — قابل تأیید نیست

نه ✅ نه ❌. با ۱۱ رویداد، `lat/lon` هویت رویداد را با `AUC = 1.0000` می‌دهد و هیچ طرح
تقسیمی این سؤال را منصفانه نمی‌پرسد.

## ۱۳ · آن سیگنال نرم منطقه‌ای چیست

**چه می‌دانم:** روی تقسیم تمیز، مکان `+4.18` می‌ارزد و این با مدل خطی هم می‌خواند.

**چه نمی‌دانم:** اقلیم است؟ بوم‌سازگان؟ الگوی کاربری زمین اروپا؟ یک متغیر پنهان دیگر؟

> "I can measure that roughly four points survive a 150 km separation. I can't yet tell you
> *what* that signal is — climate, biome, land-use policy. Decomposing it is the obvious
> follow-up."

## ۱۴ · هیچ اجرای چند-GPU یا HPC در این پروژه نیست

همه‌چیز روی یک GTX 1070. نه DDP، نه SLURM، نه اندازه‌گیری بازده مقیاس‌دهی.
**پیشینهٔ زیرساختی من دارایی واقعی است — ولی دارایی از پیشینه است، نه از این پروژه.**
این دو نباید قاطی شوند.

---

## و اگر چیزی پرسیدند که در این فهرست نیست

> "I haven't looked at that. Can I tell you how I'd approach it?"

بعد **بلند فکر کن**: چه چیزی را اندازه می‌گیری، کنترلت چیست، چه چیزی نتیجه را باطل
می‌کند. **این دقیقاً همان مهارتی است که برای دکترا می‌سنجند** — و تو در این پروژه
سیزده بار انجامش داده‌ای.

⛔ و هرگز، تحت هیچ فشاری، عددی نگو که مطمئن نیستی. **یک عدد غلط، همهٔ عددهای درستت
را هم می‌برد.**
