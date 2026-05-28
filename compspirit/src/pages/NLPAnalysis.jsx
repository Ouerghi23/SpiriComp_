// src/pages/NLPAnalysis.jsx
// CHANGES vs previous:
//   - NotificationToast component (new)
//   - notifications state (new)
//   - BroadcastChannel now reads complaint data from message
//   - location.pathname trigger for instant refresh on navigation

import { Link, useLocation }            from 'react-router-dom'
import { useState, useEffect, useMemo } from 'react'
import { useTranslation }               from 'react-i18next'
import ReactApexChart                   from 'react-apexcharts'
import {
  MessageSquare, Globe, AlertTriangle, Check, X, RefreshCw,
  ExternalLink, Trash2, Filter, ArrowUpDown, Tag, Percent,
  ChevronDown, Phone, Calendar, Wifi, User, Hash, MapPin,
  Zap, Key, Radio, Search, Bell,
} from 'lucide-react'
import { Badge, Spinner, EmptyState, baseChartOptions } from '../components/UI'
import { nlpApi } from '../api/client'

const C = {
  bg:'#080808', bg2:'#0C0C0C', bg3:'#0A0A0A',
  border:'rgba(255,255,255,.055)', text:'#F8FAFC',
  textMuted:'rgba(248,250,252,.5)', textDim:'rgba(248,250,252,.32)',
  red:'#CF0A2C', redLight:'#FF4060', blue:'#3B82F6', cyan:'#22D3EE',
  green:'#22C55E', amber:'#F59E0B', orange:'#F97316', purple:'#A855F7',
}

const SENT_COLORS  = { critique:C.red, négatif:C.amber, neutre:C.textMuted, positif:C.green }
const LANG_COLORS  = { ar:C.red, fr:C.cyan, en:C.green }
const LANG_LABELS  = { ar:'Arabic', fr:'French', en:'English' }
const STATUS_COLOR = { open:C.redLight, in_progress:C.amber, resolved:C.green }
const URGENCY_STYLE = {
  'très urgent':{ bg:'rgba(207,10,44,.12)',  color:C.redLight, border:'rgba(207,10,44,.3)'  },
  'urgent':     { bg:'rgba(245,158,11,.12)', color:'#FCD34D',  border:'rgba(245,158,11,.3)' },
  'normal':     { bg:'rgba(34,197,94,.08)',  color:C.green,    border:'rgba(34,197,94,.25)' },
}

// ══════════════════════════════════════════════════════════════════════
// NOTIFICATION TOAST
// ══════════════════════════════════════════════════════════════════════
function NotificationToast({ notif, onDismiss }) {
  const [progress, setProgress] = useState(100)
  const isComplaint = notif.is_complaint === true
  const accent      = isComplaint ? C.red   : C.blue
  const accentLight = isComplaint ? C.redLight : '#93C5FD'
  const DURATION    = 6000

  useEffect(() => {
    const start = Date.now()
    const timer = setInterval(() => {
      const pct = Math.max(0, 100 - ((Date.now() - start) / DURATION * 100))
      setProgress(pct)
      if (pct === 0) { clearInterval(timer); onDismiss(notif.id) }
    }, 60)
    return () => clearInterval(timer)
  }, [])

  const urgStyle = URGENCY_STYLE[notif.urgency_level] || URGENCY_STYLE['normal']

  return (
    <div style={{
      background:  C.bg2,
      border:      `1px solid ${accent}50`,
      boxShadow:   `0 8px 32px rgba(0,0,0,.65), 0 0 0 1px ${accent}18`,
      width:       320,
      position:    'relative',
      overflow:    'hidden',
      animation:   'toast-in .35s cubic-bezier(.22,1,.36,1)',
      flexShrink:  0,
    }}>
      {/* Top stripe */}
      <div style={{ height:2, background:`linear-gradient(90deg,${accent},${accent}40,transparent)` }}/>

      {/* Header */}
      <div style={{ padding:'10px 14px 8px', display:'flex', alignItems:'center',
        justifyContent:'space-between', borderBottom:`1px solid rgba(255,255,255,.06)` }}>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <Bell size={12} color={accentLight}/>
          <span style={{ fontSize:9, fontWeight:800, color:accentLight,
            letterSpacing:'2px', textTransform:'uppercase' }}>
            {isComplaint ? 'Nouvelle Réclamation' : 'Nouveau Feedback'}
          </span>
        </div>
        <button onClick={() => onDismiss(notif.id)} style={{ background:'none', border:'none',
          color:C.textDim, cursor:'pointer', padding:2, display:'flex', transition:'color .15s' }}
          onMouseEnter={e => e.currentTarget.style.color = C.text}
          onMouseLeave={e => e.currentTarget.style.color = C.textDim}>
          <X size={12}/>
        </button>
      </div>

      {/* Body */}
      <div style={{ padding:'10px 14px 12px' }}>
        {/* ID */}
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:20,
          fontWeight:900, color:accentLight, letterSpacing:'-.3px', marginBottom:10 }}>
          {notif.complaint_id}
        </div>

        {/* Detail grid */}
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'6px 16px' }}>
          {[
            { label:'Catégorie',  value:notif.category,         color:C.blue  },
            { label:'Urgence',    value:notif.urgency_level,     color:notif.urgency_level==='très urgent'?C.red:notif.urgency_level==='urgent'?C.amber:C.green },
            { label:'Langue',     value:notif.language_detected, color:C.cyan  },
            { label:'Sentiment',  value:notif.sentiment,         color:SENT_COLORS[notif.sentiment]||C.textMuted },
          ].map(item => (
            <div key={item.label}>
              <div style={{ fontSize:8, color:C.textDim, letterSpacing:'1.5px',
                textTransform:'uppercase', marginBottom:2, fontWeight:700 }}>{item.label}</div>
              <div style={{ fontSize:11, fontWeight:700, color:item.color,
                overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                {item.value||'—'}
              </div>
            </div>
          ))}
        </div>

        {/* City detected */}
        {notif.city_detected && (
          <div style={{ marginTop:8, fontSize:10, color:C.green,
            display:'flex', alignItems:'center', gap:4 }}>
            <MapPin size={10}/> {notif.city_detected}
          </div>
        )}

        {/* Urgency badge for complaints */}
        {isComplaint && notif.urgency_level && (
          <div style={{ marginTop:8 }}>
            <span style={{ ...urgStyle, padding:'3px 10px', fontSize:9,
              fontWeight:800, letterSpacing:'1px', textTransform:'uppercase', display:'inline-block' }}>
              {notif.urgency_level}
            </span>
          </div>
        )}
      </div>

      {/* Countdown progress bar */}
      <div style={{ height:2, background:'rgba(255,255,255,.06)' }}>
        <div style={{ height:'100%', background:accent, width:`${progress}%`,
          transition:'width .06s linear' }}/>
      </div>
    </div>
  )
}

// ── Shared sub-components ──────────────────────────────────────────────
const SectionLabel = ({ children, action, sub }) => (
  <div style={{ marginTop:40, marginBottom:16 }}>
    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
      <div style={{ fontSize:10, fontWeight:800, color:C.red, letterSpacing:'4.5px',
        textTransform:'uppercase', display:'flex', alignItems:'center', gap:12 }}>
        <span style={{ width:22, height:1, background:C.red, display:'inline-block', flexShrink:0 }}/>
        {children}
      </div>
      {action && <div style={{ flexShrink:0 }}>{action}</div>}
    </div>
    {sub && <div style={{ fontSize:10, color:C.textDim, letterSpacing:'1px', marginTop:5, paddingLeft:34 }}>{sub}</div>}
  </div>
)

const StatBlock = ({ label, value, color, icon:IconComp, sub }) => (
  <div className="nlp-stat-block" style={{ background:C.bg3, border:`1px solid ${C.border}`,
    padding:'22px 18px', position:'relative', overflow:'hidden',
    transition:'all .3s cubic-bezier(.22,1,.36,1)', cursor:'default' }}>
    <div style={{ position:'absolute', top:0, left:'12%', right:'12%', height:1,
      background:`linear-gradient(90deg,transparent,${color||C.red},transparent)` }}/>
    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:12 }}>
      <span style={{ fontSize:9, fontWeight:700, color:C.textDim, letterSpacing:'1.8px',
        textTransform:'uppercase', lineHeight:1.5 }}>{label}</span>
      {IconComp && (
        <div style={{ width:24, height:24, border:`1px solid ${(color||C.red)}30`,
          background:`${color||C.red}10`, display:'flex', alignItems:'center',
          justifyContent:'center', flexShrink:0 }}>
          <IconComp size={11} color={color||C.red}/>
        </div>
      )}
    </div>
    <div style={{ marginBottom:sub?6:0 }}>
      <span style={{ fontFamily:"'Barlow Condensed',sans-serif",
        fontSize:typeof value==='string'&&value.length>6?24:34,
        fontWeight:900, color:color||C.red, lineHeight:1, letterSpacing:'-1px' }}>{value}</span>
    </div>
    {sub && <div style={{ fontSize:9, color:C.textDim, letterSpacing:'1px', textTransform:'uppercase' }}>{sub}</div>}
  </div>
)

const ChartPanel = ({ title, sub, children, style={} }) => (
  <div className="nlp-chart-panel" style={{ background:C.bg2, border:`1px solid ${C.border}`,
    padding:'20px 22px', position:'relative', overflow:'hidden',
    transition:'border-color .3s', ...style }}>
    <div className="nlp-panel-accent" style={{ position:'absolute', top:0, left:0, right:0,
      height:'1.5px', background:`linear-gradient(90deg,transparent,${C.red},transparent)`,
      transform:'scaleX(0)', transformOrigin:'center', transition:'transform .4s ease' }}/>
    {(title||sub) && (
      <div style={{ marginBottom:14 }}>
        {title && <div style={{ fontSize:11, fontWeight:700, color:C.text, letterSpacing:'.5px', marginBottom:2 }}>{title}</div>}
        {sub   && <div style={{ fontSize:10, color:C.textDim, letterSpacing:'1px' }}>{sub}</div>}
      </div>
    )}
    {children}
  </div>
)

const StyledSelect = ({ value, onChange, options, id }) => (
  <div className="nlp-select-wrap">
    <select id={id} value={value} onChange={e=>onChange(e.target.value)} className="nlp-select">
      {options.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
    <ChevronDown size={11} color={C.textDim}/>
  </div>
)

const ActionBtn = ({ onClick, disabled, children, variant='default', small=true }) => {
  const V = {
    amber:   { bg:'rgba(245,158,11,.12)', border:'rgba(245,158,11,.3)', color:'#FCD34D' },
    green:   { bg:'rgba(34,197,94,.12)',  border:'rgba(34,197,94,.3)',  color:C.green   },
    red:     { bg:'rgba(207,10,44,.15)',  border:'rgba(207,10,44,.35)', color:C.redLight},
    redSolid:{ bg:C.red, border:'transparent', color:'#fff' },
    ghost:   { bg:'transparent', border:C.border, color:C.textMuted },
    default: { bg:'rgba(255,255,255,.04)', border:C.border, color:C.textMuted },
  }
  const v = V[variant]||V.default
  return (
    <button onClick={onClick} disabled={disabled} style={{
      background:v.bg, border:`1px solid ${v.border}`, color:v.color,
      padding:small?'5px 10px':'8px 18px', fontSize:10, fontWeight:700,
      cursor:disabled?'not-allowed':'pointer', opacity:disabled?0.5:1,
      transition:'all .2s', display:'inline-flex', alignItems:'center', gap:5,
      whiteSpace:'nowrap', fontFamily:"'Inter',system-ui", letterSpacing:'.3px' }}
      onMouseOver={e=>{ if(!disabled) e.currentTarget.style.opacity='.8' }}
      onMouseOut={e=>{  if(!disabled) e.currentTarget.style.opacity='1'  }}>
      {children}
    </button>
  )
}

function ComplaintModal({ complaint:c, onClose, onStatusUpdate, onDelete,
                          t, actionLoading, confirmDelete, setConfirmDelete }) {
  if (!c) return null
  const urgStyle    = URGENCY_STYLE[c.nlp_urgency_level]||URGENCY_STYLE['normal']
  const isActioning = actionLoading === c.complaint_id
  const isComplaint = c.is_complaint !== undefined ? c.is_complaint : null
  const statusLabel = {
    open:        t('nlp.statusOpen'),
    in_progress: t('nlp.statusInProgress'),
    resolved:    t('nlp.statusResolved'),
  }
  const keywords = Array.isArray(c.nlp_keywords) ? c.nlp_keywords : []

  return (
    <div onClick={onClose} style={{ position:'fixed', inset:0, zIndex:9000,
      background:'rgba(0,0,0,.75)', backdropFilter:'blur(4px)',
      display:'flex', alignItems:'center', justifyContent:'center', padding:24 }}>
      <div onClick={e=>e.stopPropagation()} style={{ background:C.bg2,
        border:`1px solid ${C.border}`, width:'100%', maxWidth:680,
        maxHeight:'90vh', overflowY:'auto', position:'relative' }}>
        <div style={{ position:'absolute', top:0, left:0, right:0, height:2,
          background:`linear-gradient(90deg,transparent,${C.red},transparent)` }}/>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start',
          padding:'22px 24px 18px', borderBottom:`1px solid ${C.border}` }}>
          <div>
            <div style={{ fontSize:9, color:C.red, letterSpacing:'2.5px', fontWeight:800,
              textTransform:'uppercase', marginBottom:6 }}>{t('nlp.popupTitle')}</div>
            <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:22,
              fontWeight:900, color:C.text, letterSpacing:'-.5px', lineHeight:1 }}>
              {c.complaint_id}
            </div>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            {isComplaint===true  && <Badge variant="red">{t('nlp.reclamation')}</Badge>}
            {isComplaint===false && <Badge variant="blue">{t('nlp.feedbackBadge')}</Badge>}
            <button onClick={onClose} style={{ background:'transparent',
              border:`1px solid ${C.border}`, color:C.textMuted, cursor:'pointer',
              width:28, height:28, display:'flex', alignItems:'center',
              justifyContent:'center', transition:'all .2s' }}
              onMouseOver={e=>{ e.currentTarget.style.borderColor=C.red; e.currentTarget.style.color=C.red }}
              onMouseOut={e=>{  e.currentTarget.style.borderColor=C.border; e.currentTarget.style.color=C.textMuted }}>
              <X size={12}/>
            </button>
          </div>
        </div>
        <div style={{ padding:'20px 24px', display:'flex', flexDirection:'column', gap:18 }}>
          <div style={{ background:C.bg3,
            border:`1px solid ${isComplaint?'rgba(207,10,44,.25)':'rgba(59,130,246,.25)'}`,
            padding:'16px 18px', position:'relative', overflow:'hidden' }}>
            <div style={{ position:'absolute', top:0, left:0, right:0, height:1,
              background:`linear-gradient(90deg,transparent,${isComplaint?C.red:C.blue},transparent)` }}/>
            <div style={{ fontSize:9, color:C.textDim, letterSpacing:'2px', fontWeight:800,
              textTransform:'uppercase', marginBottom:12 }}>{t('nlp.popupContact')}</div>
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
              <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                <div style={{ width:32, height:32, background:`${C.cyan}14`, border:`1px solid ${C.cyan}30`, display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
                  <Phone size={14} color={C.cyan}/>
                </div>
                <div>
                  <div style={{ fontSize:9, color:C.textDim, letterSpacing:'1.5px', textTransform:'uppercase', fontWeight:700, marginBottom:3 }}>{t('nlp.popupMsisdn')}</div>
                  <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:18, fontWeight:800, color:C.cyan, letterSpacing:'.5px' }}>
                    {c.msisdn||<span style={{ fontSize:12, color:C.textDim }}>{t('nlp.popupUnknown')}</span>}
                  </div>
                </div>
              </div>
              <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                <div style={{ width:32, height:32, background:`${C.purple}14`, border:`1px solid ${C.purple}30`, display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
                  <User size={14} color={C.purple}/>
                </div>
                <div>
                  <div style={{ fontSize:9, color:C.textDim, letterSpacing:'1.5px', textTransform:'uppercase', fontWeight:700, marginBottom:3 }}>{t('nlp.popupSegment')}</div>
                  <div style={{ fontSize:13, fontWeight:600, color:C.textMuted }}>{c.segment||t('nlp.popupUnknown')}</div>
                </div>
              </div>
            </div>
          </div>
          <div style={{ background:C.bg3, border:`1px solid ${C.border}`, padding:'14px 16px' }}>
            <div style={{ fontSize:9, color:C.textDim, letterSpacing:'2px', fontWeight:800, textTransform:'uppercase', marginBottom:10 }}>{t('nlp.popupText')}</div>
            <p style={{ fontSize:13, color:C.text, lineHeight:1.7, margin:0, wordBreak:'break-word' }}>{c.text_original}</p>
          </div>
          <div>
            <div style={{ fontSize:9, color:C.textDim, letterSpacing:'2px', fontWeight:800, textTransform:'uppercase', marginBottom:10 }}>{t('nlp.popupNlp')}</div>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:1, background:'rgba(255,255,255,.04)' }}>
              {[
                { label:t('nlp.language'),    value:LANG_LABELS[c.language]||c.language?.toUpperCase(), color:LANG_COLORS[c.language]||C.textMuted, Icon:Globe    },
                { label:t('nlp.category'),    value:c.nlp_category,  color:C.blue,  Icon:Tag  },
                { label:t('nlp.sentiment'),   value:c.nlp_sentiment, color:SENT_COLORS[c.nlp_sentiment]||C.textMuted, Icon:Zap },
                { label:t('nlp.popupScore'),  value:c.nlp_urgency_score?.toFixed(2)?? '—', color:C.amber, Icon:Percent },
                { label:t('nlp.city'),        value:c.nlp_city||t('nlp.popupUnknown'), color:C.green, Icon:MapPin },
                { label:t('nlp.popupNetwork'),value:c.nlp_network_type||t('nlp.popupUnknown'), color:C.cyan, Icon:Wifi },
              ].map(({ label, value, color, Icon })=>(
                <div key={label} style={{ background:C.bg3, padding:'12px 14px', position:'relative', overflow:'hidden' }}>
                  <div style={{ position:'absolute', top:0, left:'10%', right:'10%', height:1, background:`linear-gradient(90deg,transparent,${color},transparent)` }}/>
                  <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:6 }}>
                    <Icon size={10} color={color}/>
                    <span style={{ fontSize:9, color:C.textDim, letterSpacing:'1.5px', fontWeight:700, textTransform:'uppercase' }}>{label}</span>
                  </div>
                  <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:16, fontWeight:800, color, letterSpacing:'-.3px' }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:12 }}>
            <span style={{ fontSize:9, color:C.textDim, letterSpacing:'2px', fontWeight:800, textTransform:'uppercase' }}>{t('nlp.tableUrg')}</span>
            <span style={{ ...urgStyle, padding:'4px 12px', fontSize:9, fontWeight:800, letterSpacing:'1.5px', textTransform:'uppercase' }}>{c.nlp_urgency_level}</span>
          </div>
          {keywords.length > 0 && (
            <div>
              <div style={{ fontSize:9, color:C.textDim, letterSpacing:'2px', fontWeight:800, textTransform:'uppercase', marginBottom:8, display:'flex', alignItems:'center', gap:6 }}>
                <Key size={10} color={C.textDim}/>{t('nlp.popupKeywords')}
              </div>
              <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                {keywords.map((kw,i)=>(
                  <span key={i} style={{ fontSize:10, padding:'3px 10px', fontWeight:600, background:'rgba(255,255,255,.05)', border:`1px solid ${C.border}`, color:C.textMuted }}>{kw}</span>
                ))}
              </div>
            </div>
          )}
          <div>
            <div style={{ fontSize:9, color:C.textDim, letterSpacing:'2px', fontWeight:800, textTransform:'uppercase', marginBottom:10 }}>{t('nlp.popupMeta')}</div>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12 }}>
              {[
                { label:t('nlp.popupDate'),    value:c.submitted_at?.slice(0,16)?.replace('T',' ')||'—', Icon:Calendar },
                { label:t('nlp.popupChannel'), value:c.channel||'—', Icon:Hash },
                { label:t('nlp.status'),       value:statusLabel[c.status]||c.status, Icon:Radio, color:STATUS_COLOR[c.status] },
              ].map(({ label, value, Icon, color })=>(
                <div key={label}>
                  <div style={{ display:'flex', alignItems:'center', gap:5, marginBottom:4 }}>
                    <Icon size={10} color={C.textDim}/>
                    <span style={{ fontSize:9, color:C.textDim, letterSpacing:'1.5px', fontWeight:700, textTransform:'uppercase' }}>{label}</span>
                  </div>
                  <div style={{ fontSize:12, color:color||C.textMuted, fontWeight:600 }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ display:'flex', gap:8, paddingTop:8, borderTop:`1px solid ${C.border}`, flexWrap:'wrap' }}>
            {c.status!=='in_progress'&&c.status!=='resolved'&&(
              <ActionBtn onClick={()=>onStatusUpdate(c.complaint_id,'in_progress')} disabled={isActioning} variant="amber" small={false}>{t('nlp.enCours')}</ActionBtn>
            )}
            {c.status!=='resolved'&&(
              <ActionBtn onClick={()=>onStatusUpdate(c.complaint_id,'resolved')} disabled={isActioning} variant="green" small={false}>
                <Check size={12}/> {t('nlp.cloture')}
              </ActionBtn>
            )}
            {confirmDelete===c.complaint_id?(
              <>
                <ActionBtn onClick={()=>onDelete(c.complaint_id)} disabled={isActioning} variant="redSolid" small={false}>{t('nlp.confirmer')}</ActionBtn>
                <ActionBtn onClick={()=>setConfirmDelete(null)} variant="ghost" small={false}><X size={12}/></ActionBtn>
              </>
            ):(
              <ActionBtn onClick={()=>setConfirmDelete(c.complaint_id)} disabled={isActioning} variant="red" small={false}><Trash2 size={12}/> Delete</ActionBtn>
            )}
            <div style={{ marginLeft:'auto' }}>
              <ActionBtn onClick={onClose} variant="ghost" small={false}>{t('nlp.popupClose')}</ActionBtn>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ══════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════
export default function NLPAnalysis() {
  const { t }    = useTranslation()
  const location = useLocation()

  const [stats,          setStats]        = useState(null)
  const [complaints,     setComplaints]   = useState([])
  const [loading,        setLoading]      = useState(true)
  const [fetchError,     setFetchError]   = useState(null)
  const [apiOnline,      setApiOnline]    = useState(true)
  const [actionLoading,  setActionLoading]= useState(null)
  const [confirmDelete,  setConfirmDelete]= useState(null)
  const [selectedComp,   setSelectedComp]= useState(null)
  const [lastRefreshed,  setLastRefreshed]= useState(null)
  const [refreshTrigger, setRefreshTrigger]=useState(0)
  const [notifications,  setNotifications]=useState([])   // ← NEW

  const [filterLang,      setFilterLang]      = useState('All')
  const [filterUrgency,   setFilterUrgency]   = useState('All')
  const [filterSentiment, setFilterSentiment] = useState('All')
  const [filterType,      setFilterType]      = useState('All')
  const [searchQuery,     setSearchQuery]     = useState('')

  const fetchData = async () => {
    console.log('🔄 fetchData appelé, trigger:', refreshTrigger)
    setLoading(true); setFetchError(null)
    try {
      const [statsRes, complaintsRes] = await Promise.all([
        nlpApi.stats(),
        nlpApi.list({
          language:     filterLang      !== 'All' ? filterLang      : undefined,
          urgency:      filterUrgency   !== 'All' ? filterUrgency   : undefined,
          sentiment:    filterSentiment !== 'All' ? filterSentiment : undefined,
          is_complaint: filterType==='complaint' ? true : filterType==='feedback' ? false : undefined,
          limit: 500,
        }),
      ])
      setStats(statsRes.data)
      setComplaints(complaintsRes.data?.complaints || [])
      console.log('📋 Complaints reçus:', complaintsRes.data?.complaints?.length)
      setApiOnline(true)
      setLastRefreshed(new Date())
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      setFetchError(`API error: ${msg}`)
      setApiOnline(false)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [filterLang, filterUrgency, filterSentiment, filterType, refreshTrigger])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => setRefreshTrigger(n => n + 1), 5_000)
    return () => clearInterval(interval)
  }, [])

  // Re-fetch instantly on navigation to this page
  useEffect(() => {
    setRefreshTrigger(n => n + 1)
  }, [location.pathname])

  // BroadcastChannel — instant notification + refresh when /form submits
  useEffect(() => {
    let bc = null
    try {
      bc = new BroadcastChannel('spiricomp')
      bc.onmessage = (e) => {
        console.log('📨 BroadcastChannel reçu:', e.data)
        if (e.data?.type === 'new_complaint') {
          setRefreshTrigger(n => n + 1)
          // Show toast notification with complaint data
          if (e.data.complaint) {
            setNotifications(prev => [
              ...prev.slice(-4),   // max 5 toasts at once
              { id: Date.now(), ...e.data.complaint }
            ])
          }
        }
      }
    } catch (_) {}
    return () => { if (bc) bc.close() }
  }, [])

  const dismissNotification = (id) =>
    setNotifications(prev => prev.filter(n => n.id !== id))

  const filteredComplaints = useMemo(() => {
    if (!searchQuery.trim()) return complaints
    const q = searchQuery.toLowerCase()
    return complaints.filter(c =>
      c.complaint_id?.toLowerCase().includes(q)  ||
      c.text_original?.toLowerCase().includes(q) ||
      c.nlp_category?.toLowerCase().includes(q)  ||
      c.nlp_city?.toLowerCase().includes(q)       ||
      c.msisdn?.toLowerCase().includes(q)         ||
      c.nlp_sentiment?.toLowerCase().includes(q)
    )
  }, [complaints, searchQuery])

  const complaintCount    = stats?.complaint_count    ?? stats?.by_type?.complaint ?? 0
  const nonComplaintCount = stats?.non_complaint_count ?? stats?.by_type?.feedback  ?? 0
  const complaintRate     = (stats?.total||0)>0 ? ((complaintCount/stats.total)*100).toFixed(1) : '—'

  const handleStatusUpdate = async (complaintId, newStatus) => {
    setActionLoading(complaintId)
    try {
      await nlpApi.updateStatus(complaintId, newStatus)
      setComplaints(prev=>prev.map(c=>c.complaint_id===complaintId?{...c,status:newStatus}:c))
      setSelectedComp(prev=>prev?.complaint_id===complaintId?{...prev,status:newStatus}:prev)
    } catch (err) {
      alert(`Status update failed: ${err?.response?.data?.detail || err?.message}`)
    } finally { setActionLoading(null) }
  }

  const handleDelete = async (complaintId) => {
    setActionLoading(complaintId)
    try {
      await nlpApi.deleteComplaint(complaintId)
      setComplaints(prev=>prev.filter(c=>c.complaint_id!==complaintId))
      setConfirmDelete(null); setSelectedComp(null)
      nlpApi.stats().then(r=>setStats(r.data)).catch(()=>{})
    } catch (err) {
      alert(`Delete failed: ${err?.response?.data?.detail || err?.message}`)
    } finally { setActionLoading(null) }
  }

  if (loading && !stats) return (
    <div style={{ padding:'40px 48px' }}>
      <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:48 }}>
        <span style={{ width:6, height:6, borderRadius:'50%', background:C.cyan, display:'inline-block', animation:'nlp-pulse 1.8s infinite' }}/>
        <span style={{ fontSize:9, fontWeight:800, letterSpacing:'2.5px', textTransform:'uppercase', color:C.cyan }}>{t('common.loading')}</span>
      </div>
      <Spinner size={48}/>
    </div>
  )

  const langChart = stats?.by_language&&Object.keys(stats.by_language).length>0 ? {
    series: Object.values(stats.by_language),
    options: { ...baseChartOptions, chart:{ ...baseChartOptions?.chart, type:'donut', background:'transparent', animations:{enabled:false} },
      labels: Object.keys(stats.by_language).map(l=>LANG_LABELS[l]||l),
      colors: Object.keys(stats.by_language).map(l=>LANG_COLORS[l]||C.textMuted),
      stroke:{ width:2, colors:[C.bg2] },
      plotOptions:{ pie:{ donut:{ size:'68%', labels:{ show:true,
        value:{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:'26px', fontWeight:900, color:C.text },
        total:{ show:true, label:'Total', fontSize:'10px', color:C.textMuted, formatter:()=>String(stats.total||0) }
      }}}},
      legend:{ position:'bottom', fontSize:'11px', labels:{ colors:C.textMuted }, itemMargin:{ horizontal:8 }},
      dataLabels:{ enabled:false },
      tooltip:{ theme:'dark', y:{ formatter:v=>`${v} ${t('nlp.complaints2')}` }},
    },
  } : null

  const sentChart = stats?.by_sentiment&&Object.keys(stats.by_sentiment).length>0 ? {
    series: Object.values(stats.by_sentiment),
    options: { ...baseChartOptions, chart:{ ...baseChartOptions?.chart, type:'donut', background:'transparent', animations:{enabled:false} },
      labels: Object.keys(stats.by_sentiment),
      colors: Object.keys(stats.by_sentiment).map(s=>SENT_COLORS[s]||C.textMuted),
      stroke:{ width:2, colors:[C.bg2] },
      plotOptions:{ pie:{ donut:{ size:'68%', labels:{ show:true,
        value:{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:'26px', fontWeight:900, color:C.text },
        total:{ show:true, label:t('nlp.critiques'), fontSize:'10px', color:C.textMuted, formatter:()=>String(stats.by_sentiment?.critique||0) }
      }}}},
      legend:{ position:'bottom', fontSize:'11px', labels:{ colors:C.textMuted }, itemMargin:{ horizontal:8 }},
      dataLabels:{ enabled:false }, tooltip:{ theme:'dark', y:{ formatter:v=>`${v} items` }},
    },
  } : null

  const catChart = stats?.by_category&&Object.keys(stats.by_category).length>0 ? {
    series:[{ data:Object.values(stats.by_category) }],
    options:{ ...baseChartOptions, chart:{ ...baseChartOptions?.chart, type:'bar', background:'transparent', animations:{enabled:false} },
      plotOptions:{ bar:{ horizontal:true, borderRadius:0, barHeight:'58%', distributed:true }},
      colors: Object.keys(stats.by_category).map((_,i)=>{ const p=[C.red,'#D41F35','#DA2E3C','#E04050','#E65060','#EC6070']; return p[i%p.length] }),
      xaxis:{ categories:Object.keys(stats.by_category), labels:{ style:{ fontSize:'9px', colors:C.textMuted }}, axisBorder:{show:false}, axisTicks:{show:false}},
      yaxis:{ labels:{ style:{ fontSize:'9px', colors:C.textMuted }, maxWidth:100 }},
      dataLabels:{ enabled:true, textAnchor:'start', offsetX:8, style:{ fontSize:'9px', fontWeight:700, colors:[C.text], fontFamily:"'Barlow Condensed',sans-serif" }},
      legend:{ show:false }, grid:{ borderColor:'rgba(255,255,255,.04)', strokeDashArray:3, xaxis:{ lines:{ show:false }}},
      tooltip:{ theme:'dark', y:{ formatter:v=>`${v} ${t('nlp.complaints2')}` }},
    },
  } : null

  const typeChart = (complaintCount>0||nonComplaintCount>0) ? {
    series:[complaintCount,nonComplaintCount],
    options:{ ...baseChartOptions, chart:{ ...baseChartOptions?.chart, type:'donut', background:'transparent', animations:{enabled:false} },
      labels:[t('nlp.reclamation'),t('nlp.feedbackBadge')], colors:[C.red,C.blue],
      stroke:{ width:2, colors:[C.bg2] },
      plotOptions:{ pie:{ donut:{ size:'68%', labels:{ show:true,
        value:{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:'26px', fontWeight:900, color:C.text },
        total:{ show:true, label:'Rate', fontSize:'10px', color:C.textMuted, formatter:()=>`${complaintRate}%` }
      }}}},
      legend:{ position:'bottom', fontSize:'11px', labels:{ colors:C.textMuted }, itemMargin:{ horizontal:8 }},
      dataLabels:{ enabled:false }, tooltip:{ theme:'dark', y:{ formatter:v=>`${v} items` }},
    },
  } : null

  const FILTER_CONFIG = [
    { id:'lang',      label:t('nlp.langFilter'),    value:filterLang,      set:setFilterLang,
      options:[{ value:'All', label:t('nlp.allLang') },{ value:'ar', label:t('nlp.arabic') },{ value:'fr', label:t('nlp.french') },{ value:'en', label:t('nlp.english') }] },
    { id:'urgency',   label:t('nlp.urgencyFilter'), value:filterUrgency,   set:setFilterUrgency,
      options:[{ value:'All', label:t('nlp.allUrgency') },{ value:'très urgent', label:t('nlp.tresUrgent') },{ value:'urgent', label:t('nlp.urgentLabel') },{ value:'normal', label:t('nlp.normalLabel') }] },
    { id:'sentiment', label:t('nlp.sentFilter'),    value:filterSentiment, set:setFilterSentiment,
      options:[{ value:'All', label:t('nlp.allSentiment') },{ value:'critique', label:t('nlp.critique') },{ value:'négatif', label:t('nlp.negatif') },{ value:'neutre', label:t('nlp.neutre') },{ value:'positif', label:t('nlp.positif') }] },
    { id:'type',      label:t('nlp.typeFilter'),    value:filterType,      set:setFilterType,
      options:[{ value:'All', label:t('nlp.allTypes') },{ value:'complaint', label:t('nlp.complaint') },{ value:'feedback', label:t('nlp.feedbackType') }] },
  ]

  const kpiTiles = [
    { label:'Total',                  value:(stats?.total||0).toLocaleString(), color:C.red,      Icon:MessageSquare, sub:t('nlp.kpiTotalSub')        },
    { label:t('nlp.kpiComplaint'),    value:complaintCount.toLocaleString(),     color:C.redLight, Icon:AlertTriangle, sub:t('nlp.kpiComplaintSub')    },
    { label:t('nlp.kpiNonComplaint'), value:nonComplaintCount.toLocaleString(),  color:C.blue,     Icon:Tag,           sub:t('nlp.kpiNonComplaintSub') },
    { label:t('nlp.kpiRate'),         value:`${complaintRate}%`,                color:C.amber,    Icon:Percent,       sub:t('nlp.kpiRateSub')         },
    { label:t('nlp.kpiArabic'),       value:stats?.by_language?.ar||0,           color:C.red,      Icon:Globe,         sub:t('nlp.kpiArabicSub')       },
    { label:t('nlp.kpiFrench'),       value:stats?.by_language?.fr||0,           color:C.cyan,     Icon:Globe,         sub:t('nlp.kpiFrenchSub')       },
    { label:t('nlp.kpiEnglish'),      value:stats?.by_language?.en||0,           color:C.green,    Icon:Globe,         sub:t('nlp.kpiEnglishSub')      },
    { label:t('nlp.kpiUrgent'),       value:stats?.by_urgency_level?.['très urgent']||0, color:C.red, Icon:AlertTriangle, sub:t('nlp.kpiUrgentSub')  },
  ]

  const lastRefreshedStr = lastRefreshed
    ? lastRefreshed.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' })
    : '—'

  return (
    <div style={{ background:C.bg, minHeight:'100vh', color:C.text }}>
      <style>{`
        @keyframes nlp-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }
        @keyframes toast-in  { from{opacity:0;transform:translateX(24px)} to{opacity:1;transform:translateX(0)} }
        .nlp-stat-block:hover { border-color:rgba(207,10,44,.22)!important; background:rgba(207,10,44,.03)!important; transform:translateY(-2px); box-shadow:0 8px 24px rgba(207,10,44,.07); }
        .nlp-chart-panel:hover { border-color:rgba(207,10,44,.2)!important; }
        .nlp-chart-panel:hover .nlp-panel-accent { transform:scaleX(1)!important; }
        .nlp-table-row:hover td { background:rgba(255,255,255,.022)!important; }
        .nlp-table-row { cursor:pointer; }
        .nlp-select { appearance:none; background:${C.bg3}; color:${C.text}; border:1px solid ${C.border}; padding:7px 32px 7px 12px; font-size:11px; font-weight:600; font-family:'Inter',system-ui; letter-spacing:.5px; cursor:pointer; outline:none; transition:border-color .2s; min-width:120px; }
        .nlp-select:hover,.nlp-select:focus { border-color:rgba(207,10,44,.4); }
        .nlp-select-wrap { position:relative; display:inline-block; }
        .nlp-select-wrap svg { position:absolute; right:9px; top:50%; transform:translateY(-50%); pointer-events:none; }
        .nlp-search { background:${C.bg3}; color:${C.text}; border:1px solid ${C.border}; padding:8px 12px 8px 34px; font-size:12px; font-weight:500; font-family:'Inter',system-ui; outline:none; transition:border-color .2s, box-shadow .2s; width:260px; }
        .nlp-search:focus { border-color:rgba(207,10,44,.5); box-shadow:0 0 0 2px rgba(207,10,44,.12); }
        .nlp-search::placeholder { color:${C.textDim}; }
        .complaint-row td:first-child { border-left:2px solid ${C.red}; }
        .feedback-row  td:first-child { border-left:2px solid rgba(255,255,255,.08); }
      `}</style>

      {/* ── NOTIFICATION TOASTS — fixed top-right ─────────────────── */}
      <div style={{ position:'fixed', top:20, right:20, zIndex:9999,
        display:'flex', flexDirection:'column', gap:10, pointerEvents:'none' }}>
        {notifications.map(n => (
          <div key={n.id} style={{ pointerEvents:'all' }}>
            <NotificationToast notif={n} onDismiss={dismissNotification}/>
          </div>
        ))}
      </div>

      <ComplaintModal
        complaint={selectedComp}
        onClose={()=>{ setSelectedComp(null); setConfirmDelete(null) }}
        onStatusUpdate={handleStatusUpdate}
        onDelete={handleDelete}
        t={t} actionLoading={actionLoading}
        confirmDelete={confirmDelete} setConfirmDelete={setConfirmDelete}
      />

      <div style={{ padding:'40px 48px 80px', maxWidth:1600, margin:'0 auto' }}>

        <div style={{ borderBottom:`1px solid ${C.border}`, paddingBottom:28, marginBottom:28 }}>
          <div style={{ display:'inline-flex', alignItems:'center', gap:10, marginBottom:20 }}>
            <div style={{ display:'flex', alignItems:'center', gap:7,
              background:'rgba(34,211,238,.1)', border:'1px solid rgba(34,211,238,.28)', padding:'6px 14px' }}>
              <span style={{ width:6, height:6, borderRadius:'50%', background:C.cyan,
                display:'inline-block', animation:'nlp-pulse 2s ease-in-out infinite' }}/>
              <span style={{ fontSize:9, fontWeight:800, letterSpacing:'2.5px',
                textTransform:'uppercase', color:'#67E8F9' }}>
                {apiOnline ? t('nlp.liveBadge') : t('nlp.offlineBadge')}
              </span>
            </div>
            <span style={{ fontSize:11, color:C.textDim, letterSpacing:'1.5px' }}>{t('nlp.subtitle2')}</span>
          </div>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-end', flexWrap:'wrap', gap:20 }}>
            <div>
              <h1 style={{ fontFamily:"'Barlow Condensed',sans-serif",
                fontSize:'clamp(28px,3.5vw,54px)', fontWeight:900,
                letterSpacing:'-1.5px', lineHeight:1, color:C.text, marginBottom:8 }}>
                {t('nlp.title').split(' ').slice(0,-1).join(' ')}{' '}
                <span style={{ color:C.red, fontStyle:'italic' }}>{t('nlp.title').split(' ').slice(-1)[0]}</span>
              </h1>
              <p style={{ fontSize:13, color:C.textMuted, fontWeight:300 }}>{t('nlp.subtitle')}</p>
            </div>
            <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
              {[
                { label:apiOnline?'● Online':'● Offline', color:apiOnline?C.green:C.red, bd:apiOnline?'rgba(34,197,94,.28)':'rgba(207,10,44,.28)', bg:apiOnline?'rgba(34,197,94,.08)':'rgba(207,10,44,.08)' },
                { label:`${(stats?.total||0).toLocaleString()} Submissions`, color:C.textMuted, bd:C.border, bg:'rgba(255,255,255,.02)' },
                { label:t('nlp.multilingual'), color:C.textMuted, bd:C.border, bg:'rgba(255,255,255,.02)' },
                { label:t('nlp.autoClassify'), color:C.cyan, bd:'rgba(34,211,238,.28)', bg:'rgba(34,211,238,.06)' },
              ].map((b,i)=>(
                <span key={i} style={{ fontSize:9, fontWeight:800, letterSpacing:'1.5px',
                  textTransform:'uppercase', padding:'5px 14px',
                  border:`1px solid ${b.bd}`, background:b.bg, color:b.color }}>
                  {b.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {fetchError && (
          <div style={{ display:'flex', alignItems:'flex-start', gap:12,
            background:'rgba(239,68,68,.08)', border:'1px solid rgba(239,68,68,.3)',
            padding:'14px 20px', marginBottom:1 }}>
            <AlertTriangle size={16} color={C.red} style={{ flexShrink:0, marginTop:1 }}/>
            <div>
              <div style={{ fontSize:12, color:C.redLight, fontWeight:700, marginBottom:4 }}>
                Fetch Error
              </div>
              <code style={{ fontSize:11, color:C.textMuted }}>{fetchError}</code>
              <div style={{ marginTop:8 }}>
                <ActionBtn onClick={()=>setRefreshTrigger(n=>n+1)} variant="redSolid" small={false}>
                  <RefreshCw size={12}/> Réessayer
                </ActionBtn>
              </div>
            </div>
          </div>
        )}

        {!apiOnline && !fetchError && (
          <div style={{ display:'flex', alignItems:'flex-start', gap:12,
            background:'rgba(245,158,11,.07)', border:'1px solid rgba(245,158,11,.28)',
            padding:'14px 20px', marginBottom:1 }}>
            <AlertTriangle size={14} color={C.amber}/>
            <div>
              <div style={{ fontSize:12, color:C.amber, fontWeight:700, marginBottom:4 }}>{t('nlp.offlineBanner')}</div>
              <div style={{ fontSize:11, color:C.textMuted }}>
                {t('nlp.startServer')}{' '}
                <code style={{ background:'rgba(255,255,255,.05)', padding:'2px 8px', fontSize:10, color:C.cyan }}>
                  uvicorn src.nlp.analytics_api:app --reload --port 8000
                </code>
              </div>
            </div>
          </div>
        )}

        <SectionLabel sub={t('nlp.kpiSub')}>{t('nlp.kpiSection')}</SectionLabel>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:1, background:'rgba(255,255,255,.04)' }}>
          {kpiTiles.map((kpi,i)=>(
            <StatBlock key={i} label={kpi.label} value={kpi.value} color={kpi.color} icon={kpi.Icon} sub={kpi.sub}/>
          ))}
        </div>

        {(stats?.total||0)>0 && (
          <>
            <SectionLabel sub={t('nlp.chartsSub')}>{t('nlp.chartsSection')}</SectionLabel>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:1, background:'rgba(255,255,255,.04)' }}>
              <ChartPanel title={t('nlp.langChartTitle')} sub={t('nlp.langChartSub')}>
                {langChart ? <ReactApexChart options={langChart.options} series={langChart.series} type="donut" height={240}/> : <EmptyState icon={<Globe size={28} color="rgba(255,255,255,.18)"/>} title={t('common.noData')}/>}
              </ChartPanel>
              <ChartPanel title={t('nlp.sentChartTitle')} sub={t('nlp.sentChartSub')}>
                {sentChart ? <ReactApexChart options={sentChart.options} series={sentChart.series} type="donut" height={240}/> : <EmptyState icon={<Tag size={28} color="rgba(255,255,255,.18)"/>} title={t('common.noData')}/>}
              </ChartPanel>
              <ChartPanel title={t('nlp.catChartTitle')} sub={t('nlp.catChartSub')}>
                {catChart ? <ReactApexChart options={catChart.options} series={catChart.series} type="bar" height={240}/> : <EmptyState icon={<Filter size={28} color="rgba(255,255,255,.18)"/>} title={t('common.noData')}/>}
              </ChartPanel>
              <ChartPanel title={t('nlp.classChartTitle')} sub={t('nlp.classChartSub')}>
                {typeChart ? (
                  <ReactApexChart options={typeChart.options} series={typeChart.series} type="donut" height={240}/>
                ) : (
                  <div style={{ height:240, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:10 }}>
                    <Tag size={28} color="rgba(255,255,255,.18)"/>
                    <div style={{ fontSize:11, color:C.textDim, textAlign:'center', lineHeight:1.6 }}>
                      {t('nlp.awaitingBackend')}<br/>
                      <span style={{ fontSize:9, letterSpacing:'1px', textTransform:'uppercase', color:C.cyan }}>{t('nlp.pythonRequired')}</span>
                    </div>
                  </div>
                )}
              </ChartPanel>
            </div>
          </>
        )}

        <SectionLabel
          action={
            <div style={{ display:'flex', alignItems:'center', gap:10 }}>
              {searchQuery && <span style={{ fontSize:10, color:C.textDim }}>{filteredComplaints.length} / {complaints.length}</span>}
              <Badge variant="cyan">{filteredComplaints.length} {t('nlp.shown')}</Badge>
              <Badge variant="gray">{(stats?.total||0).toLocaleString()} {t('nlp.totalLabel')}</Badge>
            </div>
          }
          sub={t('nlp.filterSub')}
        >
          {t('nlp.filterSection')}
        </SectionLabel>

        <div style={{ display:'flex', gap:1, background:'rgba(255,255,255,.04)', marginBottom:1, flexWrap:'wrap', alignItems:'stretch' }}>
          {FILTER_CONFIG.map(f=>(
            <div key={f.id} style={{ background:C.bg3, border:`1px solid ${C.border}`, padding:'10px 16px', display:'flex', alignItems:'center', gap:10 }}>
              <span style={{ fontSize:9, fontWeight:800, color:C.textDim, letterSpacing:'2px', textTransform:'uppercase', whiteSpace:'nowrap' }}>{f.label}</span>
              <StyledSelect value={f.value} onChange={f.set} options={f.options} id={f.id}/>
            </div>
          ))}
          <div style={{ background:C.bg3, border:`1px solid ${C.border}`, padding:'10px 16px', display:'flex', alignItems:'center', gap:12 }}>
            <ActionBtn onClick={()=>setRefreshTrigger(n=>n+1)} disabled={loading} variant="redSolid" small={false}>
              <RefreshCw size={12} style={{ animation:loading?'nlp-pulse .8s infinite':undefined }}/>
              {t('nlp.refreshBtn')}
            </ActionBtn>
            <div style={{ display:'flex', flexDirection:'column', gap:1 }}>
              <span style={{ fontSize:8, color:C.textDim, letterSpacing:'1.5px', textTransform:'uppercase' }}>Dernière MAJ</span>
              <span style={{ fontSize:11, color:C.textMuted, fontFamily:"'Barlow Condensed',sans-serif", fontWeight:700 }}>{lastRefreshedStr}</span>
            </div>
          </div>
          <div style={{ marginLeft:'auto', background:C.bg3, border:`1px solid ${C.border}`, padding:'10px 16px', display:'flex', alignItems:'center' }}>
            <Link to="/form" target="_blank" rel="noreferrer"
              style={{ display:'inline-flex', alignItems:'center', gap:7, background:'transparent', color:C.textMuted, fontSize:11, fontWeight:600, textDecoration:'none', letterSpacing:'.5px', transition:'color .2s' }}
              onMouseOver={e=>e.currentTarget.style.color=C.text}
              onMouseOut={e=>e.currentTarget.style.color=C.textMuted}>
              <ExternalLink size={12}/> {t('nlp.customerForm')}
            </Link>
          </div>
        </div>

        <div style={{ background:C.bg3, border:`1px solid ${C.border}`, borderTop:'none', padding:'10px 16px', display:'flex', alignItems:'center', gap:10, marginBottom:1 }}>
          <div style={{ position:'relative', flex:1, maxWidth:420 }}>
            <Search size={13} color={C.textDim} style={{ position:'absolute', left:10, top:'50%', transform:'translateY(-50%)', pointerEvents:'none' }}/>
            <input type="text" className="nlp-search"
              placeholder="Rechercher par ID, texte, catégorie, ville, MSISDN…"
              value={searchQuery} onChange={e=>setSearchQuery(e.target.value)}/>
          </div>
          {searchQuery && (
            <button onClick={()=>setSearchQuery('')} style={{ background:'transparent', border:`1px solid ${C.border}`, color:C.textMuted, cursor:'pointer', padding:'6px 10px', fontSize:10, display:'flex', alignItems:'center', gap:4, transition:'all .15s', fontFamily:'inherit' }}
              onMouseOver={e=>{ e.currentTarget.style.borderColor=C.red; e.currentTarget.style.color=C.red }}
              onMouseOut={e=>{  e.currentTarget.style.borderColor=C.border; e.currentTarget.style.color=C.textMuted }}>
              <X size={11}/> Effacer
            </button>
          )}
          <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:6, fontSize:9, color:C.textDim, letterSpacing:'1px' }}>
            <span style={{ width:6, height:6, borderRadius:'50%', background:C.green, display:'inline-block', animation:'nlp-pulse 3s ease-in-out infinite' }}/>
            Auto-refresh 5s
          </div>
        </div>

        <div style={{ border:`1px solid ${C.border}`, overflow:'hidden', position:'relative' }}>
          <div style={{ position:'absolute', top:0, left:0, right:0, height:'1.5px', background:`linear-gradient(90deg,transparent,${C.red},transparent)` }}/>
          <div style={{ overflowX:'auto' }}>
            <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11, minWidth:1100 }}>
              <thead>
                <tr style={{ background:'rgba(255,255,255,.025)', borderBottom:`1px solid ${C.border}` }}>
                  {[
                    { label:t('nlp.tableId'),      Icon:ArrowUpDown  },
                    { label:t('nlp.tableType'),    Icon:Tag           },
                    { label:t('nlp.tableText'),    Icon:null          },
                    { label:t('nlp.tableLang'),    Icon:Globe         },
                    { label:t('nlp.tableCat'),     Icon:Filter        },
                    { label:t('nlp.tableSent'),    Icon:null          },
                    { label:t('nlp.tableUrg'),     Icon:AlertTriangle },
                    { label:t('nlp.tableScore'),   Icon:null          },
                    { label:t('nlp.tableCity'),    Icon:null          },
                    { label:t('nlp.tableStatus'),  Icon:null          },
                    { label:t('nlp.tableActions'), Icon:null          },
                  ].map(({ label, Icon })=>(
                    <th key={label} style={{ padding:'12px 12px', textAlign:'left', fontSize:9, fontWeight:800, letterSpacing:'1.5px', textTransform:'uppercase', color:C.textDim, whiteSpace:'nowrap' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:5 }}>
                        {Icon && <Icon size={9} color={C.textDim}/>}{label}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredComplaints.length === 0 ? (
                  <tr>
                    <td colSpan={11} style={{ padding:48, textAlign:'center', color:C.textMuted }}>
                      {searchQuery
                        ? <><Search size={20} color="rgba(255,255,255,.18)" style={{ display:'block', margin:'0 auto 12px' }}/>Aucun résultat pour <strong style={{ color:C.text }}>"{searchQuery}"</strong></>
                        : apiOnline ? t('nlp.noSubmissions') : t('nlp.apiOffline')
                      }
                    </td>
                  </tr>
                ) : filteredComplaints.map(c => {
                  const urgStyle    = URGENCY_STYLE[c.nlp_urgency_level]||URGENCY_STYLE['normal']
                  const isActioning = actionLoading === c.complaint_id
                  const isComplaint = c.is_complaint !== undefined ? c.is_complaint : null
                  return (
                    <tr key={c.complaint_id||c.id}
                      className={`nlp-table-row${isComplaint===true?' complaint-row':isComplaint===false?' feedback-row':''}`}
                      style={{ borderBottom:`1px solid rgba(255,255,255,.04)`, opacity:isActioning?0.5:1, transition:'all .15s' }}
                      onClick={()=>setSelectedComp(c)}>
                      <td style={{ padding:'9px 12px' }}>
                        <span style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:13, fontWeight:800, color:C.red, letterSpacing:'-.3px' }}>{c.complaint_id}</span>
                      </td>
                      <td style={{ padding:'9px 12px', whiteSpace:'nowrap' }}>
                        {isComplaint===true  && <Badge variant="red">{t('nlp.reclamation')}</Badge>}
                        {isComplaint===false && <Badge variant="blue">{t('nlp.feedbackBadge')}</Badge>}
                        {isComplaint===null  && <span style={{ fontSize:9, color:C.textDim, letterSpacing:'1.5px', textTransform:'uppercase' }}>—</span>}
                      </td>
                      <td style={{ padding:'9px 12px', color:C.text, maxWidth:180, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }} title={c.text_original}>{c.text_original}</td>
                      <td style={{ padding:'9px 12px' }}>
                        <Badge variant={c.language==='ar'?'red':c.language==='fr'?'cyan':'green'}>{c.language?.toUpperCase()}</Badge>
                      </td>
                      <td style={{ padding:'9px 12px', color:C.textMuted, maxWidth:100, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', fontSize:10 }}>{c.nlp_category}</td>
                      <td style={{ padding:'9px 12px' }}>
                        <Badge variant={c.nlp_sentiment==='critique'?'red':c.nlp_sentiment==='négatif'?'amber':c.nlp_sentiment==='positif'?'green':'gray'}>{c.nlp_sentiment}</Badge>
                      </td>
                      <td style={{ padding:'9px 12px', whiteSpace:'nowrap' }}>
                        <span style={{ ...urgStyle, padding:'3px 8px', fontSize:9, fontWeight:800, letterSpacing:'1px', textTransform:'uppercase' }}>{c.nlp_urgency_level}</span>
                      </td>
                      <td style={{ padding:'9px 12px' }}>
                        <span style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:14, fontWeight:700, color:C.textMuted }}>{c.nlp_urgency_score?.toFixed(2)??'—'}</span>
                      </td>
                      <td style={{ padding:'9px 12px', color:C.textDim, fontSize:10 }}>{c.nlp_city||'—'}</td>
                      <td style={{ padding:'9px 12px', whiteSpace:'nowrap' }}>
                        <div style={{ display:'flex', alignItems:'center', gap:7 }}>
                          <span style={{ width:6, height:6, borderRadius:'50%', background:STATUS_COLOR[c.status]||C.textDim, flexShrink:0 }}/>
                          <span style={{ fontSize:10, color:C.textMuted }}>{c.status}</span>
                        </div>
                      </td>
                      <td style={{ padding:'9px 12px' }} onClick={e=>e.stopPropagation()}>
                        <div style={{ display:'flex', gap:4, flexWrap:'nowrap' }}>
                          {c.status!=='in_progress'&&c.status!=='resolved'&&(
                            <ActionBtn onClick={()=>handleStatusUpdate(c.complaint_id,'in_progress')} disabled={isActioning} variant="amber">{t('nlp.enCours')}</ActionBtn>
                          )}
                          {c.status!=='resolved'&&(
                            <ActionBtn onClick={()=>handleStatusUpdate(c.complaint_id,'resolved')} disabled={isActioning} variant="green">
                              <Check size={10}/> {t('nlp.cloture')}
                            </ActionBtn>
                          )}
                          {confirmDelete===c.complaint_id?(
                            <>
                              <ActionBtn onClick={()=>handleDelete(c.complaint_id)} disabled={isActioning} variant="redSolid">{t('nlp.confirmer')}</ActionBtn>
                              <ActionBtn onClick={()=>setConfirmDelete(null)} variant="ghost"><X size={9}/></ActionBtn>
                            </>
                          ):(
                            <ActionBtn onClick={()=>setConfirmDelete(c.complaint_id)} disabled={isActioning} variant="red"><Trash2 size={10}/></ActionBtn>
                          )}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}