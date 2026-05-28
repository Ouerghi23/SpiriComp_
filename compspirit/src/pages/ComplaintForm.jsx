// src/pages/ComplaintForm.jsx
// CHANGE vs previous: BroadcastChannel now includes res.data
// so NLPAnalysis can show a notification without a fetch roundtrip.
// Only the handleSubmit try-block changed — everything else identical.

import { useState } from 'react'
import {
  AlertTriangle, Info, Clock, ClipboardList, CheckCircle2,
  MapPin, Globe, Tag, Zap, Send, Loader2, Phone,
} from 'lucide-react'
import { nlpApi } from '../api/client'

const C = {
  primary:'#DC143C', primaryDark:'#A50E2D',
  bg:'#08080C', surface:'#101018', surfaceHi:'#161622',
  text:'#F1F1F3', textMuted:'#8888A0', textDim:'#55556A',
  border:'rgba(255,255,255,.07)', borderFocus:'#DC143C',
  red:'#EF4444', redLight:'#FCA5A5',
  blue:'#3B82F6', blueLight:'#93C5FD',
  green:'#22C55E', amber:'#F59E0B', cyan:'#22D3EE',
}

const L = {
  fr: {
    title:'Portail Client', subtitle:'Ooredoo Tunisia — Service Client',
    sections:{ contact:'Coordonnées', account:'Compte', message:'Votre message' },
    msisdn:'Numéro MSISDN', msisdnHint:'Format: XXXXXXXX',
    msisdnError:'Numéro invalide. Format attendu: XXXXXXXX',
    city:'Ville', cityPh:'Tunis, Sfax, Sousse…',
    segment:'Segment', segmentPh:'-- Sélectionner --',
    channel:'Canal', channelWeb:'Portail Web', channelApp:'App Mobile', channelSocial:'Réseaux Sociaux',
    textLabel:'Décrivez votre problème ou demande',
    placeholder:"Ex: Mon réseau 4G coupe à Sfax depuis 3 jours…\n\nOu: Comment activer le roaming international ?",
    hint:'Langue et type détectés automatiquement', charLimit:'/ 3000',
    submit:'Envoyer', submitting:'Analyse en cours…',
    minChars:'Minimum 10 caractères requis.',
    apiError:'Erreur de connexion. Vérifiez que le serveur NLP est démarré.',
    reclamation:'Réclamation', reclamationSub:'Problème de service détecté automatiquement',
    reclamationNext:'Notre équipe traitera votre réclamation dans les meilleurs délais.',
    feedback:'Feedback', feedbackSub:"Message de retour ou demande d'information",
    feedbackNext:'Merci pour votre message. Il a bien été enregistré.',
    pending:'Classification en attente', pendingSub:"Le type n'a pas encore été déterminé.",
    ref:'Référence', lang:'Langue', cat:'Catégorie', sent:'Sentiment',
    urg:'Urgence', city2:'Ville détectée', resp:'Délai de réponse',
  },
  ar: {
    title:'بوابة العملاء', subtitle:'Ooredoo تونس — خدمة العملاء',
    sections:{ contact:'بيانات الاتصال', account:'الحساب', message:'رسالتك' },
    msisdn:'رقم الهاتف', msisdnHint:'الصيغة: XXXXXXXX',
    msisdnError:'رقم غير صحيح. الصيغة المطلوبة: XXXXXXXX',
    city:'المدينة', cityPh:'تونس، صفاقس، سوسة…',
    segment:'الشريحة', segmentPh:'-- اختر --',
    channel:'القناة', channelWeb:'البوابة الإلكترونية', channelApp:'تطبيق الجوال', channelSocial:'وسائل التواصل',
    textLabel:'صف مشكلتك أو طلبك',
    placeholder:'مثال: شبكتي مقطوعة في تونس منذ 3 أيام…',
    hint:'تُكشف اللغة والنوع تلقائياً', charLimit:'/ 3000',
    submit:'إرسال', submitting:'جاري التحليل…',
    minChars:'يجب إدخال 10 أحرف على الأقل.',
    apiError:'خطأ في الاتصال. تأكد من تشغيل خادم NLP.',
    reclamation:'شكوى', reclamationSub:'تم رصد مشكلة في الخدمة تلقائياً',
    reclamationNext:'سيتولى فريقنا معالجة شكواك في أقرب وقت ممكن.',
    feedback:'تعليق', feedbackSub:'رسالة إيجابية أو طلب معلومات',
    feedbackNext:'شكراً على رسالتك. تم تسجيلها بنجاح.',
    pending:'التصنيف قيد الانتظار', pendingSub:'لم يتم تحديد النوع بعد.',
    ref:'المرجع', lang:'اللغة', cat:'الفئة', sent:'المشاعر',
    urg:'الأولوية', city2:'المدينة المكتشفة', resp:'وقت الاستجابة',
  },
  en: {
    title:'Customer Portal', subtitle:'Ooredoo Tunisia — Customer Service',
    sections:{ contact:'Contact Details', account:'Account', message:'Your message' },
    msisdn:'MSISDN / Phone number', msisdnHint:'Format: XXXXXXXX',
    msisdnError:'Invalid number. Expected format: XXXXXXXX',
    city:'City', cityPh:'Tunis, Sfax, Sousse…',
    segment:'Segment', segmentPh:'-- Select --',
    channel:'Channel', channelWeb:'Web Portal', channelApp:'Mobile App', channelSocial:'Social Networks',
    textLabel:'Describe your issue or request',
    placeholder:'Ex: My 4G network keeps dropping in Tunis since yesterday…',
    hint:'Language and type auto-detected', charLimit:'/ 3000',
    submit:'Send', submitting:'Analyzing…',
    minChars:'Minimum 10 characters required.',
    apiError:'Connection error. Make sure the NLP server is running.',
    reclamation:'Complaint', reclamationSub:'Service problem automatically detected',
    reclamationNext:'Our team will handle your complaint as soon as possible.',
    feedback:'Feedback', feedbackSub:'Positive comment or information request',
    feedbackNext:'Thank you for your message. It has been recorded.',
    pending:'Classification pending', pendingSub:'Type has not been determined yet.',
    ref:'Reference', lang:'Language', cat:'Category', sent:'Sentiment',
    urg:'Urgency', city2:'Detected city', resp:'Response time',
  },
}

const URGENCY_STYLE = {
  'très urgent':{ bg:'rgba(239,68,68,.15)',  color:'#FCA5A5', border:'rgba(239,68,68,.3)'  },
  'urgent':     { bg:'rgba(245,158,11,.15)', color:'#FCD34D', border:'rgba(245,158,11,.3)' },
  'normal':     { bg:'rgba(34,197,94,.15)',  color:'#6EE7B7', border:'rgba(34,197,94,.3)'  },
}

function validateMsisdn(v) {
  if (!v) return true
  return /^[0-9]{8}$/.test(v.replace(/\s/g, ''))
}

function ClassificationResult({ result, labels }) {
  const rawValue = result?.is_complaint
  const state    = rawValue===true?'reclamation':rawValue===false?'feedback':'pending'
  const TOKEN = {
    reclamation:{ Icon:AlertTriangle, headline:labels.reclamation, sub:labels.reclamationSub, next:labels.reclamationNext, NextIcon:ClipboardList, accent:C.red, accentLight:C.redLight, accentBg:'rgba(239,68,68,.06)', accentBorder:'rgba(239,68,68,.25)' },
    feedback:   { Icon:Info, headline:labels.feedback, sub:labels.feedbackSub, next:labels.feedbackNext, NextIcon:CheckCircle2, accent:C.blue, accentLight:C.blueLight, accentBg:'rgba(59,130,246,.06)', accentBorder:'rgba(59,130,246,.25)' },
    pending:    { Icon:Clock, headline:labels.pending, sub:labels.pendingSub, next:'', NextIcon:null, accent:C.textMuted, accentLight:C.textMuted, accentBg:'rgba(255,255,255,.03)', accentBorder:C.border },
  }
  const tk = TOKEN[state]
  return (
    <div style={{ marginTop:20, border:`1px solid ${tk.accentBorder}`, borderRadius:12, overflow:'hidden', background:tk.accentBg, animation:'cf-slideIn .35s cubic-bezier(.22,1,.36,1)' }}>
      <div style={{ padding:'18px 20px', borderBottom:`1px solid ${tk.accentBorder}`, display:'flex', alignItems:'center', gap:16, flexWrap:'wrap' }}>
        <div style={{ width:48, height:48, flexShrink:0, background:`${tk.accent}15`, border:`2px solid ${tk.accent}35`, borderRadius:10, display:'flex', alignItems:'center', justifyContent:'center' }}>
          <tk.Icon size={22} color={tk.accentLight}/>
        </div>
        <div style={{ flex:1, minWidth:0 }}>
          <div style={{ fontFamily:"'Barlow Condensed','Inter',sans-serif", fontSize:20, fontWeight:900, color:tk.accentLight, letterSpacing:'-.3px', lineHeight:1.1, marginBottom:4, textTransform:'uppercase' }}>{tk.headline}</div>
          <div style={{ fontSize:12, color:C.textMuted, lineHeight:1.5 }}>{tk.sub}</div>
        </div>
        <div style={{ textAlign:'right', flexShrink:0 }}>
          <div style={{ fontSize:9, color:C.textMuted, letterSpacing:'1.5px', textTransform:'uppercase', marginBottom:3 }}>{labels.ref}</div>
          <div style={{ fontFamily:"'Barlow Condensed',monospace", fontSize:15, fontWeight:800, color:tk.accentLight }}>{result.complaint_id}</div>
        </div>
      </div>
      <div style={{ padding:'14px 20px', display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:'12px 20px' }}>
        <ResultDetail label={labels.lang} value={result.language_detected} Icon={Globe} color={result.language_detected==='العربية'?C.red:result.language_detected==='Français'?C.cyan:C.green}/>
        <ResultDetail label={labels.cat} value={result.category} Icon={Tag}/>
        <ResultDetail label={labels.sent} value={result.sentiment} Icon={Zap} color={result.sentiment==='critique'?C.red:result.sentiment==='négatif'?C.amber:result.sentiment==='positif'?C.green:C.textMuted}/>
        {state==='reclamation'&&result.urgency_level&&(
          <div>
            <div style={detailLabelStyle}><Clock size={9} style={{ display:'inline', marginRight:4 }}/>{labels.urg}</div>
            <span style={{ display:'inline-block', ...(URGENCY_STYLE[result.urgency_level]||URGENCY_STYLE['normal']), padding:'3px 10px', borderRadius:20, fontSize:11, fontWeight:700, letterSpacing:'.5px', textTransform:'uppercase' }}>{result.urgency_level}</span>
          </div>
        )}
        {result.city_detected&&<ResultDetail label={labels.city2} value={result.city_detected} Icon={MapPin} color={C.green}/>}
        {state==='reclamation'&&result.estimated_response_hours!=null&&<ResultDetail label={labels.resp} value={`${result.estimated_response_hours}h`} Icon={Clock} color={tk.accentLight}/>}
      </div>
      {tk.next&&(
        <div style={{ padding:'12px 20px', borderTop:`1px solid ${tk.accentBorder}`, display:'flex', alignItems:'flex-start', gap:10 }}>
          {tk.NextIcon&&<tk.NextIcon size={14} color={tk.accentLight} style={{ flexShrink:0, marginTop:2 }}/>}
          <p style={{ fontSize:12, color:C.textMuted, margin:0, lineHeight:1.65 }}>{tk.next}</p>
        </div>
      )}
    </div>
  )
}

const detailLabelStyle = { fontSize:9, color:'#8888A0', fontWeight:700, letterSpacing:'1.2px', textTransform:'uppercase', marginBottom:5, display:'flex', alignItems:'center' }

function ResultDetail({ label, value, Icon, color }) {
  return (
    <div>
      <div style={detailLabelStyle}>{Icon&&<Icon size={9} style={{ display:'inline', marginRight:4 }}/>}{label}</div>
      <div style={{ fontSize:13, fontWeight:600, color:color||'#F1F1F3' }}>{value||'—'}</div>
    </div>
  )
}

function SectionDivider({ label }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:10, margin:'22px 0 16px' }}>
      <span style={{ fontSize:9, fontWeight:800, color:'#55556A', letterSpacing:'2px', textTransform:'uppercase', whiteSpace:'nowrap' }}>{label}</span>
      <div style={{ flex:1, height:1, background:'rgba(255,255,255,.07)' }}/>
    </div>
  )
}

function getInputProps() {
  return {
    style:{ width:'100%', padding:'11px 14px', background:'#08080C', color:'#F1F1F3', border:'1px solid rgba(255,255,255,.07)', borderRadius:8, fontSize:14, outline:'none', fontFamily:'inherit', transition:'border-color .15s', boxSizing:'border-box', appearance:'none', WebkitAppearance:'none' },
    onFocus:e=>{ e.target.style.borderColor='#DC143C'; e.target.style.boxShadow='0 0 0 3px #DC143C18' },
    onBlur: e=>{ e.target.style.borderColor='rgba(255,255,255,.07)'; e.target.style.boxShadow='none' },
  }
}

const LANG_TABS = [
  { code:'fr', flag:'🇫🇷', label:'Français' },
  { code:'ar', flag:'🇹🇳', label:'عربي'    },
  { code:'en', flag:'🇬🇧', label:'English' },
]

export default function ComplaintForm() {
  const [lang,        setLang]        = useState('fr')
  const [form,        setForm]        = useState({ msisdn:'', city:'', segment:'', channel:'web', text:'' })
  const [result,      setResult]      = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)
  const [fieldErrors, setFieldErrors] = useState({})

  const labels = L[lang]||L.fr
  const inp    = getInputProps()
  const dir    = lang==='ar'?'rtl':'ltr'

  const handleLang = code => { setLang(code); setResult(null); setError(null); setFieldErrors({}) }

  const validate = () => {
    const errs = {}
    if (form.msisdn && !validateMsisdn(form.msisdn)) errs.msisdn = labels.msisdnError
    if (form.text.trim().length < 10)                 errs.text   = labels.minChars
    return errs
  }

  const handleSubmit = async e => {
    e.preventDefault()
    const errs = validate()
    if (Object.keys(errs).length > 0) { setFieldErrors(errs); return }
    setFieldErrors({})
    setLoading(true); setError(null); setResult(null)

    console.log('🚀 SUBMIT — texte envoyé:', form.text)
    console.log('🌐 API URL:', import.meta.env.VITE_API_URL || 'http://localhost:8000')

    try {
      const res = await nlpApi.submit({
        text:    form.text,
        msisdn:  form.msisdn  || null,
        city:    form.city    || null,
        segment: form.segment || null,
        channel: form.channel,
      })

      console.log('✅ RÉPONSE BACKEND:', res.data)
      console.log('🆔 Complaint ID:', res.data?.complaint_id)
      console.log('📂 Sauvegardé dans: data/nlp/complaints.db')

      setResult(res.data)
      setForm(f => ({ ...f, text:'', msisdn:'', city:'' }))

      // ── Notifie le dashboard NLP — inclut les données pour la toast ──
      try {
        new BroadcastChannel('spiricomp').postMessage({
          type:      'new_complaint',
          complaint: res.data,   // ← données complètes pour la notification
        })
        console.log('📡 BroadcastChannel envoyé → dashboard notifié')
      } catch (bcErr) {
        console.warn('⚠️ BroadcastChannel non supporté:', bcErr)
      }

      setTimeout(() => {
        document.getElementById('cf-result')?.scrollIntoView({ behavior:'smooth', block:'start' })
      }, 100)
    } catch (err) {
      console.error('❌ ERREUR SUBMIT:', err)
      console.error('❌ STATUS:', err?.response?.status)
      console.error('❌ DÉTAIL:', err?.response?.data)
      setError(labels.apiError)
    } finally {
      setLoading(false)
    }
  }

  const charCount = form.text.length
  const charColor = charCount>2800?C.amber:charCount>100?C.green:C.textDim
  const charPct   = (charCount/3000)*100

  return (
    <>
      <style>{`
        *,*::before,*::after{box-sizing:border-box}
        @keyframes cf-slideIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
        @keyframes cf-spin{to{transform:rotate(360deg)}}
        .cf-spin{animation:cf-spin .7s linear infinite}
        .cf-page{min-height:100vh;background:${C.bg};display:flex;align-items:flex-start;justify-content:center;padding:0;font-family:'Inter',system-ui,sans-serif}
        .cf-card{background:${C.surface};border:1px solid ${C.border};width:100%;max-width:560px;border-radius:0;padding:32px 28px 40px;margin:0 auto;min-height:100vh}
        .cf-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
        .cf-lang-tabs{display:flex;gap:8px;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid ${C.border};flex-wrap:wrap}
        .cf-lang-btn{flex:1;min-width:80px;padding:9px 14px;border-radius:8px;border:1.5px solid ${C.border};background:transparent;color:${C.textMuted};font-size:13px;font-weight:600;cursor:pointer;transition:all .18s;font-family:inherit;display:flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap}
        .cf-lang-btn.active{background:${C.primary};border-color:${C.primary};color:white}
        .cf-lang-btn:hover:not(.active){border-color:${C.primary}60;color:${C.text}}
        .cf-label{display:block;color:${C.textMuted};font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:7px}
        .cf-field-error{font-size:11px;color:#FCA5A5;margin-top:5px;display:flex;align-items:center;gap:4px}
        .cf-textarea{width:100%;padding:12px 14px;background:${C.bg};color:${C.text};border:1px solid ${C.border};border-radius:8px;font-size:14px;outline:none;font-family:inherit;resize:vertical;min-height:120px;line-height:1.65;transition:border-color .15s,box-shadow .15s}
        .cf-textarea:focus{border-color:${C.primary};box-shadow:0 0 0 3px ${C.primary}18}
        .cf-char-bar{height:2px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;margin-top:6px}
        .cf-char-bar-fill{height:100%;border-radius:2px;transition:width .2s,background .2s}
        .cf-submit{width:100%;padding:14px;background:${C.primary};color:white;border:none;border-radius:8px;font-size:15px;font-weight:700;cursor:pointer;transition:background .18s,transform .12s,box-shadow .18s;font-family:inherit;letter-spacing:.3px;display:flex;align-items:center;justify-content:center;gap:8px;min-height:48px;margin-top:20px}
        .cf-submit:hover:not(:disabled){background:${C.primaryDark};box-shadow:0 6px 20px ${C.primary}40;transform:translateY(-1px)}
        .cf-submit:disabled{background:#333;cursor:not-allowed}
        @media(min-width:480px){.cf-page{padding:32px 16px;align-items:center}.cf-card{min-height:auto;border-radius:16px;box-shadow:0 12px 48px rgba(0,0,0,.5)}}
        @media(min-width:768px){.cf-page{padding:48px 24px}.cf-card{padding:40px 36px 48px}}
        @media(max-width:359px){.cf-grid-2{grid-template-columns:1fr}.cf-card{padding:24px 18px 32px}.cf-lang-btn{font-size:12px;padding:8px 10px}}
      `}</style>

      <div className="cf-page" dir={dir}>
        <div className="cf-card">
          <div style={{ textAlign:'center', marginBottom:28 }}>
            <div style={{ width:54, height:54, background:'linear-gradient(135deg,#DC143C,#8B0000)', borderRadius:13, display:'inline-flex', alignItems:'center', justifyContent:'center', color:'white', fontSize:26, fontWeight:800, marginBottom:14, boxShadow:`0 6px 20px ${C.primary}40` }}>O</div>
            <h1 style={{ color:C.text, fontSize:22, fontWeight:700, margin:'0 0 5px', letterSpacing:'-.3px' }}>{labels.title}</h1>
            <p style={{ color:C.textMuted, fontSize:12, margin:0 }}>{labels.subtitle}</p>
          </div>

          <div className="cf-lang-tabs" role="tablist">
            {LANG_TABS.map(({ code, flag, label }) => (
              <button key={code} className={`cf-lang-btn${lang===code?' active':''}`}
                onClick={() => handleLang(code)} role="tab" aria-selected={lang===code}>
                <span>{flag}</span><span>{label}</span>
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <SectionDivider label={labels.sections.contact}/>
            <div className="cf-grid-2">
              <div>
                <label className="cf-label" htmlFor="cf-msisdn">
                  <Phone size={9} style={{ display:'inline', marginRight:4 }}/>{labels.msisdn}
                </label>
                <input id="cf-msisdn" type="tel" value={form.msisdn}
                  onChange={e => { setForm(f=>({...f,msisdn:e.target.value})); if(fieldErrors.msisdn) setFieldErrors(fe=>({...fe,msisdn:''})) }}
                  placeholder={labels.msisdnHint} {...inp}
                  style={{ ...inp.style, borderColor:fieldErrors.msisdn?C.red:C.border }}/>
                {fieldErrors.msisdn&&<div className="cf-field-error"><AlertTriangle size={10}/>{fieldErrors.msisdn}</div>}
                <div style={{ fontSize:10, color:C.textDim, marginTop:4 }}>{labels.msisdnHint}</div>
              </div>
              <div>
                <label className="cf-label" htmlFor="cf-city">
                  <MapPin size={9} style={{ display:'inline', marginRight:4 }}/>{labels.city}
                </label>
                <input id="cf-city" type="text" value={form.city}
                  onChange={e=>setForm(f=>({...f,city:e.target.value}))}
                  placeholder={labels.cityPh} {...inp}/>
              </div>
            </div>

            <SectionDivider label={labels.sections.account}/>
            <div className="cf-grid-2">
              <div>
                <label className="cf-label" htmlFor="cf-segment">{labels.segment}</label>
                <select id="cf-segment" value={form.segment}
                  onChange={e=>setForm(f=>({...f,segment:e.target.value}))}
                  {...inp} style={{ ...inp.style, cursor:'pointer' }}>
                  <option value="">{labels.segmentPh}</option>
                  {['Standard','Premium','Enterprise','VIP'].map(s=><option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="cf-label" htmlFor="cf-channel">{labels.channel}</label>
                <select id="cf-channel" value={form.channel}
                  onChange={e=>setForm(f=>({...f,channel:e.target.value}))}
                  {...inp} style={{ ...inp.style, cursor:'pointer' }}>
                  <option value="web">{labels.channelWeb}</option>
                  <option value="app">{labels.channelApp}</option>
                  <option value="social">{labels.channelSocial}</option>
                </select>
              </div>
            </div>

            <SectionDivider label={labels.sections.message}/>
            <div style={{ marginBottom:6 }}>
              <label className="cf-label" htmlFor="cf-text">{labels.textLabel}</label>
              <textarea id="cf-text" className="cf-textarea" value={form.text}
                onChange={e=>{ setForm(f=>({...f,text:e.target.value})); if(fieldErrors.text) setFieldErrors(fe=>({...fe,text:''})) }}
                placeholder={labels.placeholder} rows={6} maxLength={3000}
                style={{ borderColor:fieldErrors.text?C.red:undefined }}/>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginTop:6 }}>
                <span style={{ fontSize:10, color:C.textDim }}>{labels.hint}</span>
                <span style={{ fontSize:10, color:charColor, fontVariantNumeric:'tabular-nums', fontWeight:600 }}>{charCount} {labels.charLimit}</span>
              </div>
              <div className="cf-char-bar">
                <div className="cf-char-bar-fill" style={{ width:`${charPct}%`, background:charCount>2800?C.amber:charCount>50?C.primary:'rgba(255,255,255,.1)' }}/>
              </div>
              {fieldErrors.text&&<div className="cf-field-error" style={{ marginTop:6 }}><AlertTriangle size={10}/>{fieldErrors.text}</div>}
            </div>

            {error&&(
              <div style={{ background:'rgba(239,68,68,.08)', border:'1px solid rgba(239,68,68,.28)', borderRadius:8, padding:'10px 14px', color:'#FCA5A5', fontSize:12, marginBottom:4, display:'flex', alignItems:'center', gap:8 }}>
                <AlertTriangle size={13}/>{error}
              </div>
            )}

            <button type="submit" className="cf-submit" disabled={loading}>
              {loading?<><Loader2 size={16} className="cf-spin"/>{labels.submitting}</>:<><Send size={15}/>{labels.submit}</>}
            </button>
          </form>

          <div id="cf-result">
            {result&&<ClassificationResult result={result} labels={labels}/>}
          </div>
        </div>
      </div>
    </>
  )
}